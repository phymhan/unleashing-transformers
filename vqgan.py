import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import visdom

# From taming transformers
class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e # number of embeddings
        self.e_dim = e_dim # dimension of embedding
        self.beta = beta # commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (self.embedding.weight**2).sum(1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None: # reshape back to match original input shape
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()

        return z_q

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ResBlock, self).__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.convs = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        return x + self.convs(x)

class Encoder(nn.Module):
    def __init__(self, nc, nf):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nf, kernel_size=4, stride=2, padding=1), # 16x16 
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(nf),
            ResBlock(nf, nf),
            nn.BatchNorm2d(nf),
            ResBlock(nf, nf),
            nn.BatchNorm2d(nf), # necessary?
        )
    
    def forward(self, x):
        return self.encoder(x)

class Generator(nn.Module):
    def __init__(self, nc, nf):
        super().__init__()
        self.generator = nn.Sequential(
            ResBlock(nf, nf),
            nn.BatchNorm2d(nf),
            ResBlock(nf, nf),
            nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nc, kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, x):
        return self.generator(x)

class VQAutoEncoder(nn.Module):
    def __init__(self, nc, nf, ne, beta=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nf, kernel_size=4, stride=2, padding=1), # 16x16 
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(nf),
            ResBlock(nf, nf),
            nn.BatchNorm2d(nf),
            ResBlock(nf, nf),
            nn.BatchNorm2d(nf), # necessary?
        )

        self.quantize = VectorQuantizer(ne, nf, beta)

        self.generator = nn.Sequential(
            ResBlock(nf, nf),
            nn.BatchNorm2d(nf),
            ResBlock(nf, nf),
            nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nc, kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        quant, codebook_loss, _ = self.quantize(x)
        x = self.generator(quant)
        return x, codebook_loss

class Discriminator(nn.Module):
    def __init__(self, nc, nf, factor=1.0, weight=0.8):
        super().__init__()
        self.disc_factor = factor
        self.disc_weight = weight
        self.discriminator = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(nc, nf, kernel_size=4, stride=2, padding=1)), # 16x16 
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1)), # 8x8
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1)), # 4x4
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(nf*4, 1, kernel_size=4, stride=1)), # 1x1
        )
    
    def calculate_adaptive_weight(self, recon_loss, g_loss, last_layer):
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * self.disc_weight        
    
    def forward(self, x):
        return self.discriminator(x)

if __name__ == '__main__':
    vis = visdom.Visdom()
    disc_factor = 1.0
    codebook_weight = 1.0

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST('~/workspace/data', train=True, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    autoencoder = VQAutoEncoder(1, 64, 10)
    discriminator = Discriminator(1, 32)

    ae_optim = torch.optim.Adam(autoencoder.parameters())
    d_optim = torch.optim.Adam(discriminator.parameters())

    g_losses, d_losses = np.array([]), np.array([])
    global_step = 0
    for epoch in range(10000):
        for x, _ in dataloader:

            ## update autoencoder
            x_hat, codebook_loss = autoencoder(x)

            recon_loss = torch.abs(x - x_hat).mean()
            disc_loss = -discriminator(x_hat).mean()

            last_layer = autoencoder.generator[-1].weight # might have to be conv2d not convtranspose2d?
            d_weight = discriminator.calculate_adaptive_weight(recon_loss, disc_loss, last_layer=last_layer)

            g_loss = recon_loss + d_weight * disc_factor * disc_loss + codebook_weight * codebook_loss.mean()
            g_losses = np.append(g_losses, g_loss.item())

            ae_optim.zero_grad()
            g_loss.backward()
            ae_optim.step()

            ## update discriminator
            if global_step % 2  == 0:
                x_hat, _ = autoencoder(x)
                
                loss_real = F.relu(1. - discriminator(x)).mean()
                loss_fake = F.relu(1. + discriminator(x_hat)).mean()
                d_loss = disc_factor * 0.5 * (loss_real + loss_fake)

                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()

                d_losses = np.append(d_losses, d_loss.item())

            if global_step % 50 == 0:
                print(f"Step {global_step}, G Loss: {g_losses.mean():.3f}, D Loss: {d_losses.mean():.3f}")
                g_losses, d_losses = np.array([]), np.array([])
                vis.images(x.clamp(0,1)[:64], win="x", opts=dict(title="x"))
                vis.images(x_hat.clamp(0,1)[:64], win="x_hat", opts=dict(title="x_hat"))
            
            global_step += 1
        
        print("Saving model")
        torch.save(autoencoder, f"autoencoder_{epoch}.pkl")
        torch.save(discriminator, f"discriminator_{epoch}.pkl")

            



