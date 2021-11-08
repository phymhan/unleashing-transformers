import torch
import torch.nn.functional as F
import torch.distributions as dists
import numpy as np
import math
import random
from .sampler import Sampler
from tqdm import tqdm

class AbsorbingDiffusion(Sampler):
    def __init__(self, H, denoise_fn, mask_id, embedding_weight, aux_weight=0.01):
        super().__init__(H, embedding_weight=embedding_weight)

        self.num_classes = H.codebook_size
        self.latent_emb_dim = H.emb_dim
        self.shape = tuple(H.latent_shape)
        self.num_timesteps = H.diffusion_steps
        self.mask_id = mask_id
        self._denoise_fn = denoise_fn
        self.n_samples = H.batch_size
        self.loss_type = H.loss_type
        self.mask_schedule = H.mask_schedule
        self.aux_weight = aux_weight
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps+1))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps+1))
        self.register_buffer('loss_history', torch.zeros(self.num_timesteps+1))

        assert self.mask_schedule in ['random', 'fixed']
    
    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            # why do we square just to sqrt again!?!?
            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(1, self.num_timesteps+1, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt

        # try cosine noise schedule
        else:
            raise ValueError
    
    def q_sample(self, x_0, t):
        # samples q(x_t | x_0)
        # randomly set token to mask with probability t/T
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.rand_like(x_t.float()) < (t.float().unsqueeze(-1) / self.num_timesteps)
        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask
    
    def q_sample_mlm(self, x_0, t):
        # samples q(x_t | x_0)
        # fixed noise schedule, masks exactly int(t/T * latent_size) tokens
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.zeros_like(x_t).to(torch.bool)

        # TODO: offset so each n_masked_tokens is picked with equal probability
        n_masked_tokens = (t.float() / self.num_timesteps) * x_t.size(1)
        n_masked_tokens = torch.round(n_masked_tokens).to(torch.int64)
        n_masked_tokens[n_masked_tokens == 0] = 1
        ones = torch.ones_like(mask[0]).to(torch.bool).to(x_0.device)

        for idx, n_tokens_to_mask in enumerate(n_masked_tokens):
            index = torch.randperm(x_0.size(1))[:n_tokens_to_mask].to(x_0.device)
            mask[idx].scatter_(dim=0, index=index, src=ones)
        
        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask

    def _train_loss(self, x_0):
        b, device = x_0.size(0), x_0.device

        # choose what time steps to compute loss at
        t, pt = self.sample_time(b, device, 'uniform')

        # make x noisy and denoise

        if self.mask_schedule == 'random':
            x_t, x_0_ignore, mask = self.q_sample(x_0=x_0, t=t)
        elif self.mask_schedule == 'fixed':
            x_t, x_0_ignore, mask = self.q_sample_mlm(x_0=x_0, t=t)

            
        # sample p(x_0 | x_t)
        x_0_hat_logits = self._denoise_fn(x_t, t=t).permute(0,2,1)
        
        # mask out codes that are never used
        # TODO: Try training from scratch with this included. Or somehow only train with codes that appear during training.
        #if self.mask is not None:
        #    x_0_hat_logits = x_0_hat_logits + self.mask.reshape(1,-1,1)

        # Always compute ELBO for comparison purposes 
        cross_entropy_loss = F.cross_entropy(x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction='none').sum(1) 
        vb_loss =  cross_entropy_loss / t
        vb_loss = vb_loss / pt
        vb_loss = vb_loss / (math.log(2) * x_0.shape[1:].numel())      
        
        if self.loss_type == 'elbo':
            loss = vb_loss 
        elif self.loss_type == 'd3pm':
            # a reweighted variational bound that also takes into account non-masked points
            aux_loss = F.cross_entropy(x_0_hat_logits, x_0, reduction='none').mean(1)
            # not sure about aux_weighting. BUG: vb_loss is now normed (i.e. / pt) so weighting will be even more broken
            loss = vb_loss + self.aux_weight * aux_loss
        elif self.loss_type == 'normed':
            denom = mask.float().sum(1)
            denom[denom == 0] = 1 # prevent divide by 0 errors.
            loss = cross_entropy_loss / denom
        elif self.loss_type == 'new':
            # denom = mask.float().sum(1)
            # denom[denom == 0] = 1 # prevent divide by 0 errors.
            # loss = cross_entropy_loss / denom
            weight = (1 - (t / self.num_timesteps))
            loss = weight * cross_entropy_loss #+ (torch.log(weight) * weight)
            # loss = loss / torch.log(torch.tensor(2.0, device=device)) # convert to bpd
            # loss = loss / pt
            loss = loss / (math.log(2) * x_0.shape[1:].numel()) 
        elif self.loss_type == 'new-normed':
            denom = mask.float().sum(1)
            denom[denom == 0] = 1 # prevent divide by 0 errors.
            loss = cross_entropy_loss / denom
            weight = (1 - (t / self.num_timesteps))
            loss = weight * loss #+ (torch.log(weight) * weight)
            # loss = loss / torch.log(torch.tensor(2.0, device=device)) # convert to bpd
            # loss = loss / pt
            # loss = loss / (math.log(2) * x_0.shape[1:].numel())    
        else:
            raise ValueError

        # Track loss at each time step history for bar plot
        Lt2_prev = self.loss_history.gather(dim=0, index=t)
        new_loss_history = (0.1 * loss + 0.9 * Lt2_prev).detach().to(self.loss_history.dtype)

        self.loss_history.scatter_(dim=0, index=t, src=new_loss_history)

        # Track loss at each time step for importance sampling
        Lt2 = vb_loss.detach().clone().pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach().to(self.loss_history.dtype)
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2).to(self.loss_history.dtype))

        return loss.mean(), vb_loss.mean()
    
    def sample(self, sample_stride='all', temp=1.0, sample_steps=None):
        b, device = self.n_samples, 'cuda'
        x_0 = torch.ones((b, np.prod(self.shape)), device=device).long() * self.mask_id

        if sample_stride == 'all':
            sample_steps = list(range(1, self.num_timesteps+1))
        elif sample_stride == 'even':
            sample_steps = np.linspace(1,self.num_timesteps, num=sample_steps).astype(np.long)
        elif sample_stride == 'quadratic':
            sample_steps = [x**2 for x in range(1, int(np.sqrt(self.num_timesteps)))]
        elif sample_stride == 'dynamic':
            sample_steps = sample_steps

        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            x_t, _, _ = self.q_sample(x_0, t)
            x_0_logits = self._denoise_fn(x_t, t=t)
            if self.mask is not None:
                x_0_logits = x_0_logits + self.mask.reshape(1,1,-1)
            # scale by temperature
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(
                logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_0[x_t == self.mask_id] = x_0_hat[x_t == self.mask_id]
            # print("x0 at", t, x_0, x_0.shape)

        return x_0
    
    def sample_v2(self, sample_stride='all', temp=1.0, sample_steps=None):
        b, device = self.n_samples, 'cuda'
        x_t = torch.ones((b, np.prod(self.shape)), device=device).long() * self.mask_id

        if sample_stride == 'all':
            sample_steps = list(range(1, self.num_timesteps+1))
        elif sample_stride == 'even':
            sample_steps = np.linspace(1,self.num_timesteps, num=sample_steps).astype(np.long)
        elif sample_stride == 'quadratic':
            sample_steps = [x**2 for x in range(1, int(np.sqrt(self.num_timesteps)))]
        elif sample_stride == 'dynamic':
            sample_steps = sample_steps
        elif sample_stride == 'magic':
            sample_steps = list(range(1, sample_steps+1))

        unmasked = torch.zeros_like(x_t, device=device).bool()

        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)

            # where to unmask
            changes = torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1)
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            # x_t, _, _ = self.q_sample(x_0, t)
            x_0_logits = self._denoise_fn(x_t, t=t)
            # if self.mask is not None:
            #     x_0_logits = x_0_logits + self.mask.reshape(1,1,-1)
            # scale by temperature
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(
                logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_t[changes] = x_0_hat[changes]
            # print("x0 at", t, x_0, x_0.shape)

        return x_t
    
    @torch.no_grad()
    def elbo_step_unweighted(self, x_0, t):
        b, device = x_0.size(0), x_0.device
        t = torch.full((b,), t, device=device, dtype=torch.long)
        x_t, x_0_ignore, _ = self.q_sample(x_0, t)
        x_0_hat_logits = self._denoise_fn(x_t, t=t).permute(0,2,1)
        cross_entropy_loss = F.cross_entropy(x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction='none').sum(1) 
        return cross_entropy_loss.mean()
    
    @torch.no_grad()
    def elbo(self, x_0):
        b, device = x_0.size(0), x_0.device
        elbo = 0.0
        for t in reversed(list(range(1, self.num_timesteps+1))):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            x_t, x_0_ignore, mask = self.q_sample(x_0=x_0, t=t)
            x_0_hat_logits = self._denoise_fn(x_t, t=t).permute(0,2,1)
            cross_entropy_loss = F.cross_entropy(x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction='none').sum(1) 
            elbo += cross_entropy_loss / t
        return elbo

    def train_iter(self, x):
        loss, vb_loss = self._train_loss(x)
        stats = {'loss': loss, 'vb_loss': vb_loss}
        return stats

    def sample_shape(self, shape, num_samples, time_steps=1000, step=1):
        device = 'cuda'
        x_t = torch.ones((num_samples,) + shape, device='cuda').long() * self.mask_id
        # time_steps = np.linspace(self.num_timesteps, 1, num=time_steps, endpoint=False, dtype=np.int)
        x_lim, y_lim = shape[0] - self.shape[1], shape[1] - self.shape[2]

        unmasked = torch.zeros_like(x_t, device=device).bool()

        autoregressive_step = 0
        for t in tqdm(list(reversed(list(range(1, time_steps+1))))):
            t = torch.full((num_samples,), t, device='cuda', dtype=torch.long)

            unmasking_method = 'autoregressive'
            if unmasking_method == 'random':
                # where to unmask
                changes = torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1).unsqueeze(-1)
                # don't unmask somewhere already unmasked
                changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
                # update mask with changes
                unmasked = torch.bitwise_or(unmasked, changes)
            elif unmasking_method == 'autoregressive':
                changes = torch.zeros(x_t.shape, device=device).bool()
                index = (int(autoregressive_step / shape[1]), autoregressive_step % shape[1])
                changes[:, index[0], index[1]] = True
                unmasked = torch.bitwise_or(unmasked, changes)
                autoregressive_step += 1

            # # make x_0 noisy
            # x_t, _, mask = self.q_sample(x_0.reshape(x_0.size(0), -1), t)
            # x_t = x_t.reshape(x_t.size(0), shape[0], shape[1])
            # mask = mask.reshape(x_t.size(0), shape[0], shape[1])
            # keep track of PoE probabilities
            x_0_probs = torch.zeros((num_samples,) + shape + (self.codebook_size,), device='cuda')
            # keep track of counts
            count = torch.zeros((num_samples,) + shape, device='cuda')

            # TODO: Monte carlo approximate this instead
            for i in range(0, x_lim+1, step):
                for j in range(0, y_lim+1, step):
                    # collect local noisy area
                    x_t_part = x_t[:,i:i+self.shape[1], j:j+self.shape[2]]

                    # increment count
                    count[:,i:i+self.shape[1], j:j+self.shape[2]] += 1.0

                    # flatten
                    x_t_part = x_t_part.reshape(x_t_part.size(0), -1)
                    
                    # denoise
                    x_0_logits_part = self._denoise_fn(x_t_part, t=t)

                    # unflatten
                    x_0_logits_part = x_0_logits_part.reshape(x_t_part.size(0), self.shape[1], self.shape[2], -1)

                    # multiply probabilities
                    # for mixture
                    x_0_probs[:,i:i+self.shape[1], j:j+self.shape[2]] += torch.softmax(x_0_logits_part, dim=-1)
                    # for PoE
                    # x_0_probs[:,i:i+self.shape[1], j:j+self.shape[2]] += torch.log_softmax(x_0_logits_part, dim=-1)

            # TODO: Try both PoE and Mixture with new sampling

            # Normalize probabilities
            
            # Product of Experts -ish (with count division probably same as a mixture)
            # temp = 4.0
            # temp = 1.0
            # x_0_probs = torch.softmax((x_0_probs / count.unsqueeze(-1)) / temp, dim=-1)
            # x_0_probs = torch.softmax((x_0_probs / temp), dim=-1)

            # Mixture with Temperature
            x_0_probs = x_0_probs / x_0_probs.sum(-1, keepdim=True)
            temp = 0.8
            C = torch.tensor(x_0_probs.size(-1)).float()
            x_0_probs = torch.softmax((torch.log(x_0_probs) + torch.log(C)) / temp, dim=-1)

            x_0_dist = dists.Categorical(probs=x_0_probs)
            x_0_hat = x_0_dist.sample().long()

            # update x_0 where anything has been masked
            x_t[changes] = x_0_hat[changes]

        return x_t