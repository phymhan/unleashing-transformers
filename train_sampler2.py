from numpy.core.fromnumeric import mean
import torch
import numpy as np
import copy
import time
import os
from tqdm import tqdm
from models import VQAutoEncoder, Generator
# from hparams import get_sampler_hparams
from utils.data_utils import get_data_loaders, cycle
from utils.sampler_utils import generate_latent_ids, get_latent_loaders, retrieve_autoencoder_components_state_dicts,\
    get_samples, get_sampler
from utils.train_utils import EMA, optim_warmup
from utils.log_utils import log, config_log, print_args, \
    save_stats, load_stats, save_model, load_model, save_images, \
    display_images, start_training_log, set_up_wandb, seed_everything
# torch.backends.cudnn.benchmark = True
import argparse
import shutil
import wandb
from pathlib import Path
import pdb
st = pdb.set_trace

def main(H, logger):
    vis = logger

    latents_fp_suffix = '_flipped' if H.horizontal_flip else ''
    latents_filepath = f'latents/{H.dataset}_{H.latent_shape[-1]}_train_latents{latents_fp_suffix}'

    train_with_validation_dataset = False
    if H.steps_per_eval:
        train_with_validation_dataset = True

    if not os.path.exists(latents_filepath):
        ae_state_dict = retrieve_autoencoder_components_state_dicts(
            H, ['encoder', 'quantize', 'generator']
        )
        ae = VQAutoEncoder(H)
        ae.load_state_dict(ae_state_dict, strict=False)
        # val_loader will be assigned to None if not training with validation dataest
        train_loader, val_loader = get_data_loaders(
            H.dataset,
            H.img_size,
            H.batch_size,
            drop_last=False,
            shuffle=False,
            get_flipped=H.horizontal_flip,
            get_val_dataloader=train_with_validation_dataset
        )

        log("Transferring autoencoder to GPU to generate latents...")
        ae = ae.cuda()  # put ae on GPU for generating
        generate_latent_ids(H, ae, train_loader, val_loader)
        log("Deleting autoencoder to conserve GPU memory...")
        ae = ae.cpu()
        ae = None

    train_latent_loader, val_latent_loader = get_latent_loaders(H, get_validation_loader=train_with_validation_dataset)

    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['quantize', 'generator'],
        remove_component_from_key=True
    )

    embedding_weight = quanitzer_and_generator_state_dict.pop(
        'embedding.weight')
    if H.deepspeed:
        embedding_weight = embedding_weight.half()
    embedding_weight = embedding_weight.cuda()
    generator = Generator(H)

    generator.load_state_dict(quanitzer_and_generator_state_dict, strict=False)
    generator = generator.cuda()
    sampler = get_sampler(H, embedding_weight).cuda()

    optim = torch.optim.Adam(sampler.parameters(), lr=H.lr)

    if H.ema:
        ema = EMA(H.ema_beta)
        ema_sampler = copy.deepcopy(sampler)

    # initialise before loading so as not to overwrite loaded stats
    losses = np.array([])
    val_losses = np.array([])
    elbo = np.array([])
    val_elbos = np.array([])
    mean_losses = np.array([])
    start_step = 0
    log_start_step = 0
    if H.load_step > 0:
        start_step = H.load_step + 1

        sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir).cuda()
        if H.ema:
            # if EMA has not been generated previously, recopy newly loaded model
            try:
                ema_sampler = load_model(
                    ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir)
            except Exception:
                ema_sampler = copy.deepcopy(sampler)
        if H.load_optim:
            optim = load_model(
                optim, f'{H.sampler}_optim', H.load_step, H.load_dir)
            # only used when changing learning rates and reloading from checkpoint
            for param_group in optim.param_groups:
                param_group['lr'] = H.lr

        try:
            train_stats = load_stats(H, H.load_step)
        except Exception:
            train_stats = None

        if train_stats is not None:
            losses, mean_losses, val_losses, elbo, H.steps_per_log

            losses = train_stats["losses"],
            mean_losses = train_stats["mean_losses"],
            val_losses = train_stats["val_losses"],
            val_elbos = train_stats["val_elbos"]
            elbo = train_stats["elbo"],
            H.steps_per_log = train_stats["steps_per_log"]
            log_start_step = 0

            losses = losses[0]
            mean_losses = mean_losses[0]
            val_losses = val_losses[0]
            val_elbos = val_elbos[0]
            elbo = elbo[0]
        else:
            log('No stats file found for loaded model, displaying stats from load step only.')
            log_start_step = start_step

    scaler = torch.cuda.amp.GradScaler()
    train_iterator = cycle(train_latent_loader)
    # val_iterator = cycle(val_latent_loader)

    log(f"Sampler params total: {sum(p.numel() for p in sampler.parameters())}")

    pbar = range(start_step, H.train_steps)
    pbar = tqdm(pbar, dynamic_ncols=True)

    for step in pbar:
        step_start_time = time.time()
        # lr warmup
        if H.warmup_iters:
            if step <= H.warmup_iters:
                optim_warmup(H, step, optim)

        x = next(train_iterator)
        x = x.cuda()

        if H.amp:
            optim.zero_grad()
            with torch.cuda.amp.autocast():
                stats = sampler.train_iter(x)

            scaler.scale(stats['loss']).backward()
            scaler.step(optim)
            scaler.update()
        else:
            stats = sampler.train_iter(x)

            if torch.isnan(stats['loss']).any():
                log(f'Skipping step {step} with NaN loss', False)
                continue
            optim.zero_grad()
            stats['loss'].backward()
            optim.step()

        losses = np.append(losses, stats['loss'].item())

        if step % H.steps_per_log == 0:
            step_time_taken = time.time() - step_start_time
            stats['step_time'] = step_time_taken
            mean_loss = np.mean(losses)
            stats['mean_loss'] = mean_loss
            mean_losses = np.append(mean_losses, mean_loss)
            losses = np.array([])

            logger.log(
                {
                    "loss": mean_loss,
                    "step": step,
                }
            )
            pbar.set_description(
                f"loss: {stats['loss']:.4f}; vb_loss: {stats['vb_loss']:.4f}"
            )

            if H.sampler == 'absorbing':
                elbo = np.append(elbo, stats['vb_loss'].item())
                data = [[label, val] for (label, val) in zip(list(range(sampler.loss_history.size(0))), sampler.loss_history)]
                table = wandb.Table(data=data, columns=["label", "value"])
                logger.log(
                    {
                        "loss_bar": wandb.plot.bar(table, "label", "value"),
                    }
                )
                logger.log(
                    {
                        "elbo": stats['vb_loss'].item(),
                        "step": step,
                    }
                )

        if H.ema and step % H.steps_per_update_ema == 0 and step > 0:
            ema.update_model_average(ema_sampler, sampler)

        images = None
        if not H.no_display_output and step % H.steps_per_display_output == 0 and step > 0:
            images = get_samples(H, generator, ema_sampler if H.ema else sampler)
            logger.log(
                {
                    f"{H.sampler}_samples": wandb.Image(images),
                    "step": step,
                }
            )

        if step % H.steps_per_save_output == 0 and step > 0:
            if images is None:
                images = get_samples(H, generator, ema_sampler if H.ema else sampler)
            save_images(images, 'samples', step, H.log_dir, H.save_individually)

        if H.steps_per_eval and step % H.steps_per_eval == 0 and step > 0:
            # calculate validation loss
            valid_loss, valid_elbo, num_samples = 0.0, 0.0, 0
            eval_repeats = 5
            log("Evaluating")
            for _ in range(eval_repeats):
                for x in val_latent_loader:
                    with torch.no_grad():
                        stats = sampler.train_iter(x.cuda())
                        valid_loss += stats['loss'].item()
                        if H.sampler == 'absorbing':
                            valid_elbo += stats['vb_loss'].item()
                        num_samples += x.size(0)
            valid_loss = valid_loss / num_samples
            if H.sampler == 'absorbing':
                valid_elbo = valid_elbo / num_samples

            val_losses = np.append(val_losses, valid_loss)
            val_elbos = np.append(val_elbos, valid_elbo)
            logger.log(
                {
                    "val_loss": valid_loss,
                    "step": step,
                }
            )
            if H.sampler == 'absorbing':
                logger.log(
                    {
                        "val_elbo": valid_elbo,
                        "step": step,
                    }
                )

        if step % H.steps_per_checkpoint == 0 and step > H.load_step:
            save_model(sampler, H.sampler, step, H.log_dir)
            save_model(optim, f'{H.sampler}_optim', step, H.log_dir)

            if H.ema:
                save_model(ema_sampler, f'{H.sampler}_ema', step, H.log_dir)

            train_stats = {
                'losses': losses,
                'mean_losses': mean_losses,
                'val_losses': val_losses,
                'elbo': elbo,
                'val_elbos': val_elbos,
                'steps_per_log': H.steps_per_log,
                'steps_per_eval': H.steps_per_eval,
            }
            save_stats(H, train_stats, step)


if __name__ == '__main__':
    from hparams.defaults.sampler_defaults import HparamsAbsorbing, HparamsAutoregressive
    from hparams.defaults.vqgan_defaults import HparamsVQGAN
    from hparams.set_up_hparams import apply_parser_values_to_H

    parser = argparse.ArgumentParser()

    # training hyperparameters
    parser.add_argument("--amp", const=True, action="store_const", default=False)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--custom_dataset_path", type=str)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ema_beta", type=float, default=0.995)
    parser.add_argument("--ema", const=True, action="store_const", default=False)
    parser.add_argument("--load_dir", type=str, default="test")
    parser.add_argument("--load_optim", const=True, action="store_const", default=False)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--steps_per_update_ema", type=int, default=10)
    parser.add_argument("--train_steps", type=int, default=100000000)

    # logging hyperparameters
    parser.add_argument("--log_dir", type=str, default="test")
    parser.add_argument("--save_individually", const=True, action="store_const", default=False)
    parser.add_argument("--steps_per_checkpoint", type=int, default=25000)
    parser.add_argument("--steps_per_display_output", type=int, default=5000)
    parser.add_argument("--steps_per_eval", type=int, default=0)
    parser.add_argument("--steps_per_log", type=int, default=10)
    parser.add_argument("--steps_per_save_output", type=int, default=5000)
    # parser.add_argument("--visdom_port", type=int, default=8097)
    # parser.add_argument("--visdom_server", type=str)

    # vqgan hyperparameters
    parser.add_argument('--attn_resolutions', nargs='+', type=int)
    parser.add_argument('--base_lr', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--ch_mult', nargs='+', type=int)
    parser.add_argument('--codebook_size', type=int)
    parser.add_argument('--diff_aug', const=True, action='store_const', default=False)
    parser.add_argument('--disc_layers', type=int)
    parser.add_argument('--disc_start_step', type=int)
    parser.add_argument('--disc_weight_max', type=float)
    parser.add_argument('--emb_dim', type=int)
    parser.add_argument('--gumbel_kl_weight', type=float)
    parser.add_argument('--gumbel_straight_through', const=True, action='store_const', default=False)
    parser.add_argument('--horizontal_flip', const=True, action='store_const', default=False)
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--latent_shape', nargs='+', type=int)
    parser.add_argument('--n_channels', type=int)
    parser.add_argument('--ndf', type=int)
    parser.add_argument('--nf', type=int)
    parser.add_argument('--perceptual_weight', type=int)
    parser.add_argument('--quantizer', type=str, choices=["nearest", "gumbel"])
    parser.add_argument('--res_blocks', type=int)

    # sampler hyperparameters
    parser.add_argument("--ae_load_dir", type=str, required=True)
    parser.add_argument("--ae_load_step", type=int, required=True)
    parser.add_argument("--attn_pdrop", type=float)
    parser.add_argument("--bert_n_emb", type=int)
    parser.add_argument("--bert_n_head", type=int)
    parser.add_argument("--bert_n_layers", type=int)
    parser.add_argument("--block_size", type=int)
    parser.add_argument("--embd_pdrop", type=float)
    parser.add_argument("--greedy_epochs", type=int)
    parser.add_argument("--greedy", const=True, action="store_const", default=False)
    parser.add_argument("--loss_type", type=str)
    parser.add_argument("--mask_schedule", type=str)
    parser.add_argument("--resid_pdrop", type=float)
    parser.add_argument("--sample_block_size", type=int)
    parser.add_argument("--sample_type", type=str, choices=["diffusion", "mlm", "maskgit"])
    parser.add_argument("--sampler", type=str, required=True, choices=["absorbing", "autoregressive"])
    parser.add_argument("--total_steps", type=int)
    parser.add_argument("--sample_steps", type=int)
    parser.add_argument("--temp", type=float)
    parser.add_argument("--warmup_iters", type=int)

    # additonal arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_root", type=str, default="logs")
    parser.add_argument("--wandb_project", type=str, default="unleashing")
    parser.add_argument("--no_display_output", action='store_true')
    parser.add_argument("--time_schedule", type=str, default="uniform")
    parser.add_argument("--sample_time_schedule", type=str, default="linear")
    parser.add_argument("--sample_with_confidence", action="store_true", help="sample with confidence, for maskgit sampler")

    # Get sampler H from parser
    parser_args = parser.parse_args()

    seed_everything(parser_args.seed)

    # has to be in this order to overwrite duplicate defaults such as batch_size and lr
    H = HparamsVQGAN(parser_args.dataset)
    H.vqgan_batch_size = H.batch_size  # used for generating samples and latents

    if parser_args.sampler == "absorbing":
        H_sampler = HparamsAbsorbing(parser_args.dataset)
    elif parser_args.sampler == "autoregressive":
        H_sampler = HparamsAutoregressive(parser_args.dataset)
    H.update(H_sampler)  # overwrites old (vqgan) H.batch_size
    H = apply_parser_values_to_H(H, parser_args)

    print_args(parser, H, True)
    shutil.copyfile('models/absorbing_diffusion.py', Path('logs') / H.log_dir / 'absorbing_diffusion.py.txt')

    logger = set_up_wandb(H)

    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')
    start_training_log(H)

    main(H, logger)
