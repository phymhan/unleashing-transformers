from .base import HparamsBase

class HparamsVQGAN(HparamsBase):
    def __init__(self, dataset):
        super().__init__(dataset)
        
        if self.dataset == 'mnist':
            self.attn_resolutions = [8]
            self.batch_size = 128
            self.ch_mult = [1,1,1]
            self.codebook_size = 10
            self.disc_layers = 1
            self.disc_max_weight = 1000
            self.disc_start_step = 2001
            self.emb_dim = 64
            self.img_size = 32
            self.latent_shape = [1, 8, 8]
            self.n_channels = 1
            self.ndf = 16
            self.nf = 32
            self.perceptual_weight = 0.0
            self.res_blocks = 1

        elif self.dataset == 'cifar10':
            self.attn_resolutions = [8]
            self.batch_size = 128
            self.ch_mult = [1,1,2]
            self.codebook_size = 128
            self.disc_layers = 1
            self.disc_max_weight = 1
            self.disc_start_step = 30001
            self.emb_dim = 256
            self.img_size = 32
            self.latent_shape = [1, 8, 8]
            self.n_channels = 3
            self.ndf = 32
            self.nf = 64
            self.perceptual_weight = 1.0
            self.res_blocks = 1

        elif self.dataset == 'flowers':
            self.attn_resolutions = [8]
            self.batch_size = 128
            self.ch_mult = [1,1,2]
            self.codebook_size = 128
            self.disc_layers = 1
            self.disc_max_weight = 1000
            self.disc_start_step = 10001
            self.emb_dim = 256
            self.img_size = 32
            self.latent_shape = [1, 8, 8]
            self.n_channels = 3
            self.ndf = 32
            self.nf = 64
            self.perceptual_weight = 1.0
            self.res_blocks = 1

        elif self.dataset == 'churches':
            self.attn_resolutions = [16]
            self.batch_size = 3
            self.ch_mult = [1, 1, 2, 2, 4]
            self.codebook_size = 1024
            self.disc_layers = 3
            self.disc_max_weight = 1000
            self.disc_start_step = 30001
            self.emb_dim = 256
            self.img_size = 256
            self.latent_shape = [1, 16, 16]
            self.n_channels = 3
            self.ndf = 64
            self.nf = 128
            self.perceptual_weight = 1.0
            self.res_blocks = 2

        elif self.dataset == 'celeba' or self.dataset == 'ffhq':

            self.attn_resolutions = [16]
            self.batch_size = 3
            self.ch_mult = [1, 1, 2, 2, 4]
            self.codebook_size = 1024
            self.disc_layers = 3
            self.disc_max_weight = 1000
            self.disc_start_step = 30001
            self.emb_dim = 256
            self.img_size = 256
            self.latent_shape = [1, 16, 16]
            self.n_channels = 3
            self.ndf = 64
            self.nf = 128
            self.perceptual_weight = 1.0
            self.res_blocks = 2

        else:
            raise KeyError(f'Defaults not defined for VQGAN model on dataset: {self.dataset}')


def add_vqgan_args(parser):

    # defaults that are same for all datasets
    parser.add_argument('--base_lr', type=float, default=4.5e-6)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--diff_aug', const=True, action='store_const', default=False)
    parser.add_argument('--gumbel_kl_weight', type=float, default=1e-8)
    parser.add_argument('--gumbel_straight_through', const=True, action='store_const', default=False)
    parser.add_argument('--quantizer', type=str, default='nearest')
    parser.add_argument('--steps_per_calc_fid', type=int, default=1000)

    # dataset-dependent arguments (do not set defaults)
    parser.add_argument('--attn_resolutions', nargs='+', type=int)
    parser.add_argument('--ch_mult', nargs='+', type=int)
    parser.add_argument('--codebook_size', type=int)
    parser.add_argument('--disc_layers', type=int)
    parser.add_argument('--disc_max_weight', type=float)
    parser.add_argument('--disc_start_step', type=int)
    parser.add_argument('--emb_dim', type=int)
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--latent_shape', nargs='+', type=int)
    parser.add_argument('--n_channels', type=int)
    parser.add_argument('--ndf', type=int)
    parser.add_argument('--nf', type=int)
    parser.add_argument('--perceptual_weight', type=int)
    parser.add_argument('--res_blocks', type=int)