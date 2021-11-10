import torch
import time

from torch.nn.functional import embedding
from models import Generator
from utils.log_utils import log, load_model, save_images
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, latent_ids_to_onehot
from tqdm import tqdm
from train_sampler import get_sampler


@torch.no_grad()
def generate_images_from_latents(H, all_latents, embedding_weight, generator):
    all_latents = all_latents.cuda()
    generator = generator.cuda()

    for idx, latents in tqdm(list(enumerate(torch.split(all_latents, H.batch_size)))):
        latents_one_hot = latent_ids_to_onehot(latents, H.latent_shape, H.codebook_size).cuda()
        q = torch.matmul(latents_one_hot, embedding_weight).view(
            latents_one_hot.size(0), H.latent_shape[1], H.latent_shape[2], H.emb_dim
        ).permute(0, 3, 1, 2).contiguous()
        gen_images = generator(q)
        # vis.images(gen_images[:64].clamp(0,1), win="FID_sample_check", opts=dict(title="FID_sample_check"))
        save_images(gen_images.detach().cpu(), "sample", idx, H.log_dir, save_indivudally=True)
    # generator = generator.cpu()
    del generator


@torch.no_grad()
def generate_latents(H, sampler):
    if H.stepping is not None:
        sample_stride, sample_steps = H.stepping.split("-")
        if sample_stride == "even":
            sample_steps = int(sample_steps)
        elif sample_stride == "magic":
            sample_steps = int(sample_steps)

    log(f"Sampling with temperature {H.temp}")
    all_latents = []
    for _ in tqdm(range(int(H.n_samples/H.batch_size))):
        if H.sampler == "absorbing":
            if H.sample_type == "v1":
                latents = sampler.sample(temp=H.temp, sample_stride=sample_stride, sample_steps=sample_steps)
            elif H.sample_type == "v2":
                latents = sampler.sample_v2(temp=H.temp, sample_stride=sample_stride, sample_steps=sample_steps)
        else:
            latents = sampler.sample(temp=H.temp)

        all_latents.append(latents.cpu())

    # all_latents = [torch.load(f"logs/{image_dir}/latents_backup_{i}.pkl") for i in range(10)]
    all_latents = torch.cat(all_latents, dim=0)
    timestamp = int(time.time())

    log(f"Saving latents to src/_pkl_files/latents_backup_{H.dataset}_{timestamp}.pkl")
    torch.save(all_latents, f"src/_pkl_files/latents_backup_{H.dataset}_{timestamp}.pkl")
    return all_latents


def get_generator_and_embedding_weight(H):
    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ["quantize", "generator"],
        remove_component_from_key=True
    )
    embedding_weight = quanitzer_and_generator_state_dict.pop("embedding.weight")
    embedding_weight = embedding_weight.cuda()    
    generator = Generator(H)
    generator.load_state_dict(quanitzer_and_generator_state_dict)
    return generator, embedding_weight


def get_sampler_and_generator(H):
    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ["quantize", "generator"],
        remove_component_from_key=True
    )
    embedding_weight = quanitzer_and_generator_state_dict.pop("embedding.weight")
    embedding_weight = embedding_weight.cuda()
    sampler = get_sampler(H, embedding_weight).cuda()

    generator = Generator(H)
    generator.load_state_dict(quanitzer_and_generator_state_dict)

    if H.load_step > 0:
        sampler = load_model(sampler, f"{H.sampler}_ema", H.load_step, H.load_dir).cuda()

    sampler = sampler.eval()
    return sampler, generator


@torch.no_grad()
def generate_samples(H):
    generator, embedding_weight = get_generator_and_embedding_weight(H)
    sampler = get_sampler(H, embedding_weight).cuda()
    if H.load_step > 0:
        sampler = load_model(sampler, f"{H.sampler}_ema", H.load_step, H.load_dir).cuda()
    else:
        raise ValueError("No load step provided, cannot load sampler")
    sampler = sampler.eval()
    all_latents = generate_latents(H, sampler)
    embedding_weight = sampler.embedding_weight.cuda().clone()
    del sampler

    generate_images_from_latents(H, all_latents, embedding_weight, generator)
