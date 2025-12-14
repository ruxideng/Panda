import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import clip
import math
import numpy as np

from .utils import encode_text_single, configure_model, UnimodalCLIP, encode_text
from .TTABase import TTABase

from .shuffle import batch_hedge_v6_images

@torch.jit.script
def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        assert closure is not None, "SAM requires a closure"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class SAR(TTABase):
    def __init__(self, clip_model, class_names, cfg):
        super(SAR, self).__init__()
        clip_model.eval()
        self.clip_model = clip_model
        self.style = cfg.get('style', 'default')

        template = 'a photo of a {}.'
        configure_model(self.clip_model, freeze_text_encoder=True, freeze_image_encoder=False)
        with torch.no_grad():
            self.clip_weights = encode_text_single(clip_model, class_names, template)

        self.uni_model = UnimodalCLIP(clip_model, self.clip_weights)
        self.num_classes, self.feat_dim = self.clip_weights.shape
        self.grad_dim = sum(p.numel() for p in self.clip_model.parameters() if p.requires_grad)

        self.cfg = cfg
        self.beta = cfg['beta']

        if self.style == 'Panda':
            self.patch_size = cfg.get('Panda', {}).get('patch_size', 32)

        self.optimizer = SAM(self.uni_model.parameters(), optim.Adam, self.cfg['lr'])
        self.copy_model_and_optimizer()
        self.steps = 1
        self.margin_e0 = cfg['e_margin_multiplier'] * math.log(self.num_classes)
        self.reset_constant_em = cfg['reset_constant_em']
        self.ema = None

    def forward(self, images):
        for _ in range(self.steps):
            if self.style == 'default':
                # SAR without Panda
                outputs, ema, reset_flag = self._forward_and_adapt_default(images)
            elif self.style == 'Panda':
                outputs, ema, reset_flag = self._forward_and_adapt_batchhedgev6(images)
            else:
                raise ValueError(f"Unknown style: {self.style}")

            if reset_flag:
                self.reset()
            self.ema = ema

        return outputs

    def _forward_and_adapt_default(self, x):
        optimizer = self.optimizer
        margin = self.margin_e0
        ema = self.ema

        optimizer.zero_grad()
        outputs = self.uni_model(x)
        entropys = softmax_entropy(outputs)
        idx = torch.where(entropys < margin)
        loss = entropys[idx].mean(0)
        loss.backward()

        optimizer.first_step(zero_grad=True)
        outputs2 = self.uni_model(x)
        ent2 = softmax_entropy(outputs2)[idx]
        loss2 = ent2[torch.where(ent2 < margin)].mean(0)
        if not np.isnan(loss2.item()):
            ema = update_ema(ema, loss2.item())
        loss2.backward()
        optimizer.second_step(zero_grad=True)

        reset_flag = ema is not None and ema < self.reset_constant_em
        return outputs, ema, reset_flag

    @torch.enable_grad()
    def _forward_and_adapt_batchhedgev6(self, x):
        optimizer = self.optimizer
        margin = self.margin_e0
        ema = self.ema
        beta = self.beta
        patch_size = self.patch_size

        optimizer.zero_grad()
        outputs = self.uni_model(x)  # (B, K)
        # gain negative augmentation images
        x_bh6 = batch_hedge_v6_images(x, patch_size)  # (M, C, H, W)
        outputs_bh6 = self.uni_model(x_bh6)  # (M, K)

        # gain average bias offset feature
        mean_logits = outputs_bh6.mean(dim=0, keepdim=True)  # (1, K)
        mean_logits = mean_logits.expand(outputs.size(0), -1)  # (B, K)

        hedge_out = outputs - beta * mean_logits  # (B, K)
        entropys = softmax_entropy(hedge_out)  # (B,)
        idx = torch.where(entropys < margin)
        loss = entropys[idx].mean(0)
        loss.backward()

        optimizer.first_step(zero_grad=True)

        outputs2 = self.uni_model(x)  # (B, K)
        x_bh6_2 = batch_hedge_v6_images(x, patch_size)  # (M, C, H, W)
        outputs_bh6_2 = self.uni_model(x_bh6_2)  # (M, K)

        mean_logits2 = outputs_bh6_2.mean(dim=0, keepdim=True)  # (1, K)
        mean_logits2 = mean_logits2.expand(outputs2.size(0), -1)  # (B, K)

        # predict images using debiased feature
        hedge2 = outputs2 - beta * mean_logits2  # (B, K)
        ent2 = softmax_entropy(hedge2)[idx]
        loss2 = ent2[torch.where(ent2 < margin)].mean(0)
        if not np.isnan(loss2.item()):
            ema = update_ema(ema, loss2.item())
        loss2.backward()

        optimizer.second_step(zero_grad=True)

        reset_flag = ema is not None and ema < self.reset_constant_em
        return hedge_out, ema, reset_flag

    def reset(self):
        self.load_model_and_optimizer()
        self.ema = None