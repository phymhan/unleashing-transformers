import math
import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
from tqdm import tqdm
from .sampler import Sampler
import pdb
st = pdb.set_trace


class AbsorbingDiffusion(Sampler):
    def __init__(self, H, denoise_fn, mask_id, embedding_weight, aux_weight=0.01):
        super().__init__(H, embedding_weight=embedding_weight)

        self.num_classes = H.codebook_size
        self.latent_emb_dim = H.emb_dim
        self.shape = tuple(H.latent_shape)
        self.num_timesteps = H.total_steps
        # self.block_size = H.block_size

        self.mask_id = mask_id
        self._denoise_fn = denoise_fn
        self.n_samples = H.batch_size
        self.time_schedule = H.time_schedule
        self.loss_type = H.loss_type
        self.mask_schedule = H.mask_schedule
        self.aux_weight = aux_weight
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps+1))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps+1))
        self.register_buffer('loss_history', torch.zeros(self.num_timesteps+1))
        self.register_buffer('mce_history', torch.zeros(self.num_timesteps+1))  # mean cross entropy loss

        assert self.mask_schedule in ['random', 'fixed']

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(1, self.num_timesteps+1, (b,), device=device).long()  # inclusive [1, T]
            pt = torch.ones_like(t).float() / self.num_timesteps  # uniform 1/T
            return t, pt
        
        # ========== MaskGIT style training ==========
        elif method == 'cosine':
            r = torch.rand(b, device=device)  # [0, 1]
            gamma_r = torch.cos(r * np.pi / 2)  # [0, 1]
            t = torch.round(gamma_r * (self.num_timesteps - 1) + 1).long()
            pr = torch.ones_like(t).float() / self.num_timesteps  # TODO: pt is not used for MaskGIT
            return t, pr
        
        elif method == 'linear':
            r = torch.rand(b, device=device)  # [0, 1]
            gamma_r = 1 - r  # [0, 1]
            t = torch.round(gamma_r * (self.num_timesteps - 1) + 1).long()
            pr = torch.ones_like(t).float() / self.num_timesteps  # TODO: pt is not used for MaskGIT
            return t, pr
        
        elif method == 'square':
            r = torch.rand(b, device=device)  # [0, 1]
            gamma_r = 1 - torch.pow(r, 2)  # [0, 1]
            t = torch.round(gamma_r * (self.num_timesteps - 1) + 1).long()
            pr = torch.ones_like(t).float() / self.num_timesteps  # TODO: pt is not used for MaskGIT
            return t, pr
        
        elif method == 'cubic':
            r = torch.rand(b, device=device)  # [0, 1]
            gamma_r = 1 - torch.pow(r, 3)  # [0, 1]
            t = torch.round(gamma_r * (self.num_timesteps - 1) + 1).long()
            pr = torch.ones_like(t).float() / self.num_timesteps  # TODO: pt is not used for MaskGIT
            return t, pr
        
        elif method == 'square_root':
            r = torch.rand(b, device=device)  # [0, 1]
            gamma_r = 1 - torch.sqrt(r)  # [0, 1]
            t = torch.round(gamma_r * (self.num_timesteps - 1) + 1).long()
            pr = torch.ones_like(t).float() / self.num_timesteps  # TODO: pt is not used for MaskGIT
            return t, pr
        
        elif method == 'square_root_one_minus':
            r = torch.rand(b, device=device)  # [0, 1]
            gamma_r = torch.sqrt(1 - r)  # [0, 1]
            t = torch.round(gamma_r * (self.num_timesteps - 1) + 1).long()
            pr = torch.ones_like(t).float() / self.num_timesteps  # TODO: pt is not used for MaskGIT
            return t, pr
        
        elif method.startswith('range_'):
            lb = float(method[6:].split('_')[0])
            ub = float(method[6:].split('_')[1])
            r = torch.rand(b, device=device)  # [0, 1]
            gamma_r = lb + (ub - lb) * r
            t = torch.round(gamma_r * (self.num_timesteps - 1) + 1).long()
            pr = torch.ones_like(t).float() / self.num_timesteps  # TODO: pt is not used for MaskGIT
            return t, pr
        
        # ========== MaskGIT style training ==========

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
        t, pt = self.sample_time(b, device, self.time_schedule)

        # make x noisy and denoise

        if self.mask_schedule == 'random':
            x_t, x_0_ignore, mask = self.q_sample(x_0=x_0, t=t)
        elif self.mask_schedule == 'fixed':
            x_t, x_0_ignore, mask = self.q_sample_mlm(x_0=x_0, t=t)

        # sample p(x_0 | x_t)
        x_0_hat_logits = self._denoise_fn(x_t, t=t).permute(0, 2, 1)  # denoise fn is a transformer

        # Always compute ELBO for comparison purposes
        cross_entropy_loss = F.cross_entropy(x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction='none').sum(1)
        vb_loss = cross_entropy_loss / t
        vb_loss = vb_loss / pt
        vb_loss = vb_loss / (math.log(2) * x_0.shape[1:].numel())
        if self.loss_type == 'elbo':
            loss = vb_loss
        elif self.loss_type == 'mlm':
            denom = mask.float().sum(1)
            denom[denom == 0] = 1  # prevent divide by 0 errors.
            loss = cross_entropy_loss / denom
        elif self.loss_type == 'reweighted_elbo':
            weight = (1 - (t / self.num_timesteps))
            loss = weight * cross_entropy_loss
            loss = loss / (math.log(2) * x_0.shape[1:].numel())
        elif self.loss_type == 'simple':
            loss = cross_entropy_loss / (math.log(2) * x_0.shape[1:].numel())
        elif self.loss_type == 'reweighted_t':
            weight = (t + 1) / self.num_timesteps
            loss = weight * cross_entropy_loss
            loss = loss / (math.log(2) * x_0.shape[1:].numel())
        else:
            raise ValueError

        # Track loss at each time step history for bar plot
        Lt2_prev = self.loss_history.gather(dim=0, index=t)
        new_loss_history = (0.1 * loss + 0.9 * Lt2_prev).detach().to(self.loss_history.dtype)
        self.loss_history.scatter_(dim=0, index=t, src=new_loss_history)

        # Track mean cross entropy loss at each time step history for bar plot
        denom = mask.float().sum(1)
        denom[denom == 0] = 1  # prevent divide by 0 errors.
        mce_loss = cross_entropy_loss / denom
        mce_prev = self.mce_history.gather(dim=0, index=t)
        new_mce_history = (0.1 * mce_loss + 0.9 * mce_prev).detach().to(self.mce_history.dtype)
        self.mce_history.scatter_(dim=0, index=t, src=new_mce_history)

        # Track loss at each time step for importance sampling
        Lt2 = vb_loss.detach().clone().pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach().to(self.loss_history.dtype)
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2).to(self.loss_history.dtype))

        return loss.mean(), vb_loss.mean()

    def sample(self, temp=1.0, sample_steps=None):
        b, device = self.n_samples, 'cuda'
        x_t = torch.ones((b, np.prod(self.shape)), device=device).long() * self.mask_id
        unmasked = torch.zeros_like(x_t, device=device).bool()
        ts_equiv = np.arange(1, sample_steps+1) * self.num_timesteps / sample_steps
        sample_steps = list(range(1, sample_steps+1))
        for t in reversed(sample_steps):  # T to 1
            tt = torch.full((b,), ts_equiv[t-1], device=device, dtype=torch.long)
            # print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)

            # where to unmask
            changes = torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1)
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            x_0_logits = self._denoise_fn(x_t, t=tt)
            # scale by temperature
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_t[changes] = x_0_hat[changes]

        return x_t
    
    def sample_maskgit(self, temp=1.0, sample_steps=None, time_schedule='linear', use_confidence=False):
        def predict_from_logits(logits, argmax=False):
            if argmax:
                probabilities = F.softmax(logits, dim=-1)
                probabilities, predictions = torch.max(probabilities, dim=-1)
            else:
                probabilities = F.softmax(logits, dim=-1)
                x_dist = dists.Categorical(probs=probabilities)
                predictions = x_dist.sample().long()
                probabilities = torch.gather(probabilities, -1, predictions.unsqueeze(-1)).squeeze(-1)
            return predictions, probabilities

        b, device = self.n_samples, 'cuda'
        T = sample_steps
        N = np.prod(self.shape)
        ts = np.arange(0, T + 1)
        if time_schedule == 'linear':
            ns = 1 - ts / T
        elif time_schedule == 'cosine':
            ns = np.cos(ts / T * np.pi / 2)
        else:
            raise NotImplementedError
        ts_equiv = np.round(ns * self.num_timesteps)
        ns = np.round(ns * N)

        x_t = torch.ones((b, N), device=device).long() * self.mask_id
        unmasked = torch.zeros_like(x_t, device=device).bool()
        one_probabilities = torch.ones((b, N), device=device) + 0.01

        for t in range(T):
            tt = torch.full((b,), ts_equiv[t], device=device, dtype=torch.long)
            # predict x_0_hat from x_t
            x_0_logits = self._denoise_fn(x_t, t=tt)
            x_0_hat, x_0_prob = predict_from_logits(x_0_logits / temp)

            if use_confidence:
                confidences = torch.where(unmasked, one_probabilities, x_0_prob)
            else:
                random_probabilities = torch.rand(x_0_prob.shape, device=device)
                confidences = torch.where(unmasked, one_probabilities, random_probabilities)
            if t == T-1:
                thresholds = 0
            else:
                thresholds = torch.topk(confidences, int(N - ns[t+1]), -1)[0][..., [-1]]

            # where to unmask
            changes = confidences >= thresholds  # changes (newly unmasked) are with high confidence
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))

            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)
            x_t[changes] = x_0_hat[changes]
        
        assert (x_t == self.mask_id).sum() == 0  # no mask_id in the sequence

        return x_t

    def sample_mlm(self, temp=1.0, sample_steps=None):
        b, device = self.n_samples, 'cuda'
        x_0 = torch.ones((b, np.prod(self.shape)), device=device).long() * self.mask_id
        sample_steps = np.linspace(1, self.num_timesteps, num=sample_steps).astype(np.long)

        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            x_t, _, _ = self.q_sample(x_0, t)
            x_0_logits = self._denoise_fn(x_t, t=t)
            # scale by temperature
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(
                logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_0[x_t == self.mask_id] = x_0_hat[x_t == self.mask_id]

        return x_0

    @torch.no_grad()
    def elbo(self, x_0):
        b, device = x_0.size(0), x_0.device
        elbo = 0.0
        for t in reversed(list(range(1, self.num_timesteps+1))):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            x_t, x_0_ignore, _ = self.q_sample(x_0=x_0, t=t)
            x_0_hat_logits = self._denoise_fn(x_t, t=t).permute(0, 2, 1)
            cross_entropy_loss = F.cross_entropy(x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction='none').sum(1)
            elbo += cross_entropy_loss / t
        return elbo

    def train_iter(self, x):
        loss, vb_loss = self._train_loss(x)
        stats = {'loss': loss, 'vb_loss': vb_loss}
        return stats

    def sample_shape(self, shape, num_samples, time_steps=1000, step=1, temp=0.8):
        device = 'cuda'
        x_t = torch.ones((num_samples,) + shape, device=device).long() * self.mask_id
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

            # keep track of PoE probabilities
            x_0_probs = torch.zeros((num_samples,) + shape + (self.codebook_size,), device='cuda')
            # keep track of counts
            count = torch.zeros((num_samples,) + shape, device='cuda')

            # TODO: Monte carlo approximate this instead
            for i in range(0, x_lim+1, step):
                for j in range(0, y_lim+1, step):
                    # collect local noisy area
                    x_t_part = x_t[:, i:i+self.shape[1], j:j+self.shape[2]]

                    # increment count
                    count[:, i:i+self.shape[1], j:j+self.shape[2]] += 1.0

                    # flatten
                    x_t_part = x_t_part.reshape(x_t_part.size(0), -1)

                    # denoise
                    x_0_logits_part = self._denoise_fn(x_t_part, t=t)

                    # unflatten
                    x_0_logits_part = x_0_logits_part.reshape(x_t_part.size(0), self.shape[1], self.shape[2], -1)

                    # multiply probabilities
                    # for mixture
                    x_0_probs[:, i:i+self.shape[1], j:j+self.shape[2]] += torch.softmax(x_0_logits_part, dim=-1)

            # Mixture with Temperature
            x_0_probs = x_0_probs / x_0_probs.sum(-1, keepdim=True)
            C = torch.tensor(x_0_probs.size(-1)).float()
            x_0_probs = torch.softmax((torch.log(x_0_probs) + torch.log(C)) / temp, dim=-1)

            x_0_dist = dists.Categorical(probs=x_0_probs)
            x_0_hat = x_0_dist.sample().long()

            # update x_0 where anything has been masked
            x_t[changes] = x_0_hat[changes]

        return x_t
