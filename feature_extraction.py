from utils.data_utils import get_data_loader
import lpips
import torch
from models import Generator
from hparams import get_sampler_hparams
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts,\
                                get_samples 
from utils.log_utils import log, setup_visdom, config_log, start_training_log, \
                             load_model, save_images, display_images
from train_sampler import get_sampler
import os
import imageio
from tqdm import tqdm
import torchvision

from torch_fidelity.utils import create_feature_extractor
import torchvision.models as models
import torch.functional as F

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

class Distance:
    def __init__(self):
        super().__init__()
        self.feat_extractor = create_feature_extractor('inception-v3-compat', ['2048']).cuda()
        self.distance_metric = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    def extract_feats(self, in0):
        feats = self.feat_extractor.forward(in0)[0]
        return feats
    
    def compare_images(self, feats0, feats1):
        # diff = (feats0-feats1)**2
        # return diff.mean(dim=-1)
        return self.distance_metric(feats0, feats1)

class FeatExtractor:
    def __init__(self):
        super().__init__()
        self.feat_extractor = models.vgg16(pretrained=True).cuda().eval()
    
    def extract_feats(self, in0):
        in0 = F.interpolate(in0, size=(224,224))
        # feats = self.feat_extractor.forward(in0)[0]
        before_fc = self.vgg16.features(in0)
        before_fc = before_fc.view(-1, 7 * 7 * 512)
        feature = self.vgg16.classifier[:4](before_fc)
        print(feature.shape) # should be 4096D
        return feature

class BigDataset(torch.utils.data.Dataset):
    def __init__(self, folder, length=None):
        self.folder = folder
        self.image_paths = os.listdir(folder)
        self.length = length if length is not None else len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = imageio.imread(self.folder+path)
        img = torch.from_numpy(img).permute(2,0,1) # -> channels first
        # How does torchvision save quantize?
        return img

    def __len__(self):
        return self.length

class NoClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, length=None):
        self.dataset = dataset
        self.length = length if length is not None else len(dataset)
    
    def __getitem__(self, index):
        return self.dataset[index][0].mul(255).clamp_(0, 255).to(torch.uint8)
    
    def __len__(self):
        return self.length

def main(H, vis):
    image_dir = H.FID_images_dir
    assert image_dir is not None

    data = 'stylegan_bedroom'

    if data == 'ours':
        images = BigDataset(f"logs/{image_dir}/images/", length=50000)
    if data == 'ours_bedroom':
        images = BigDataset(f"logs/absorbing-bedrooms-new-sampling-256steps/images/", length=50000)
    if data == 'ours_ffhq':
        images = BigDataset(f"logs/absorbing-ffhq-samples/images/", length=50000)
    elif data == 'stylegan2':
        images = BigDataset(f"../stylegan2-ada-pytorch-orig/out/", length=50000)
    elif data == 'stylegan2-ffhq':
        images = BigDataset(f"../stylegan2-ada-pytorch-orig/out-ffhq/", length=50000)
    elif data == 'stylegan_bedroom':
        images = torchvision.datasets.ImageFolder(f"../stylegan/bedroom/",  transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.ToTensor()
        ]))
        images = NoClassDataset(images, length=50000)
    elif data == 'churches':
        dataset = torchvision.datasets.LSUN('../../../data/LSUN', classes=['church_outdoor_train'], transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop(256),
                    torchvision.transforms.ToTensor()
                ]))
        images = NoClassDataset(dataset, length=50000)
    elif data == 'bedroom':
        dataset = torchvision.datasets.LSUN('/projects/cgw/lsun', classes=['bedroom_train'], transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop(256),
                    torchvision.transforms.ToTensor()
                ]))
        images = NoClassDataset(dataset, length=50000)
    elif data == 'ffhq':
        dataset = torchvision.datasets.ImageFolder('../../../data/FFHQ-256',  transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.ToTensor()
        ]))
        images = NoClassDataset(dataset, length=50000)
    elif data == 'vdvae':
        images = BigDataset(f"../vdvae/out/", length=50000)
    elif data == 'progan_churches':
        images = BigDataset(f"../proGAN/churches/", length=50000)
    elif data == 'progan_bedroom':
        images = BigDataset(f"../proGAN/bedroom/211-lsun-bedroom-256x256/211-lsun-bedroom-256x256/", length=50000)
    elif data == 'vqgan_ffhq':
        images = BigDataset(f"../taming-transformers/logs/ffhq/samples/top_k_1024_temp_1.00_top_p_1.0/13750/", length=50000)
    elif data == 'vqgan_churches':
        images = BigDataset(f"logs/autoregressive-churches-flip-samples/images/", length=50000)
    else:
        raise Exception('Wrong data')
    image_dataloader = torch.utils.data.DataLoader(images, batch_size=128)

    all_feats = []
    distance_fn = Distance()

    for batch in tqdm(image_dataloader):
        batch = batch.cuda()
        feats = distance_fn.extract_feats(batch)
        all_feats.append(feats.cpu())

    torch.save(all_feats, f'all_feats_{data}.pkl')


    # import torch
    # import numpy as np
    # from prdc import compute_prdc

    # real_features = torch.cat(torch.load('all_feats_dataset.pkl'), dim=0)[:50000].numpy()
    # fake_features = torch.cat(torch.load('all_feats_stylegan2.pkl'), dim=0)[:50000].numpy()
    # print(real_features.shape, fake_features.shape)
    # nearest_k = 3

    # metrics = compute_prdc(real_features=real_features,
    #                     fake_features=fake_features,
    #                     nearest_k=nearest_k)

    # print(metrics)


if __name__=='__main__':
    H = get_sampler_hparams()
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')   
    start_training_log(H)
    main(H, vis)