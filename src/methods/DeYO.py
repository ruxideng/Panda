import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import clip
import math
import torchvision
from einops import rearrange
from .utils import encode_text_single, configure_model, UnimodalCLIP, encode_text
from .TTABase import TTABase

from .shuffle import batch_hedge_v6_images

@torch.jit.script
def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.enable_grad()
def forward_and_adapt_default(x, model, optimizer, margin, reset_constant_em, ema, cfg):
    logits = model(x)
    optimizer.zero_grad()
    entropys = softmax_entropy(logits)
    if cfg.get('filter_ent', True):
        filter_ids_1 = torch.where(entropys < cfg.get('deyo_margin', margin))[0]
    else:
        filter_ids_1 = torch.where(entropys <= math.log(cfg.get('num_classes', 1000)))[0]
    if len(filter_ids_1) == 0:
        return logits, ema, False
    x_prime = x[filter_ids_1].detach().clone()
    aug_type = cfg.get('aug_type', 'none')
    if aug_type == 'occ':
        occ_size = cfg.get('occlusion_size', 32)
        row_start = cfg.get('row_start', 0)
        col_start = cfg.get('column_start', 0)
        patch_mean = x_prime.view(x_prime.size(0), x_prime.size(1), -1).mean(dim=2, keepdim=True).view(x_prime.size(0), x_prime.size(1), 1, 1)
        window = patch_mean.expand(-1, -1, occ_size, occ_size)
        x_prime[:, :, row_start:row_start+occ_size, col_start:col_start+occ_size] = window
    elif aug_type == 'patch':
        patch_len = cfg.get('patch_len', 4)
        resize_t = torchvision.transforms.Resize(((x.shape[-1] // patch_len) * patch_len,) * 2)
        resize_o = torchvision.transforms.Resize((x.shape[-1], x.shape[-1]))
        x_prime = resize_t(x_prime)
        x_prime = rearrange(x_prime, 'b c (ph h) (pw w) -> b (ph pw) c h w', ph=patch_len, pw=patch_len)
        perm_idx = torch.argsort(torch.rand(x_prime.shape[0], x_prime.shape[1]), dim=-1)
        x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1), perm_idx]
        x_prime = rearrange(x_prime, 'b (ph pw) c h w -> b c (ph h) (pw w)', ph=patch_len, pw=patch_len)
        x_prime = resize_o(x_prime)
    elif aug_type == 'pixel':
        x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
        x_prime = x_prime[:, :, torch.randperm(x_prime.shape[-1])]
        x_prime = rearrange(x_prime, 'b c (h w) -> b c h w', h=x.shape[-2], w=x.shape[-1])
    with torch.no_grad():
        outputs_prime = model(x_prime)
    prob = logits[filter_ids_1].softmax(1)
    prob_prime = outputs_prime.softmax(1)
    cls1 = prob.argmax(dim=1)
    plpd = torch.gather(prob, 1, cls1.view(-1, 1)) - torch.gather(prob_prime, 1, cls1.view(-1, 1))
    plpd = plpd.view(-1)
    if cfg.get('filter_plpd', True):
        filter_ids_2 = torch.where(plpd > cfg.get('plpd_threshold', 0.0))[0]
    else:
        filter_ids_2 = torch.where(plpd >= -2.0)[0]
    if len(filter_ids_2) == 0:
        return logits, ema, False
    entropys = entropys[filter_ids_1][filter_ids_2]
    plpd = plpd[filter_ids_2]
    if cfg.get('reweight_ent', 0.0) > 0 or cfg.get('reweight_plpd', 0.0) > 0:
        coeff = (
            cfg.get('reweight_ent', 0.0) * (1.0 / torch.exp(entropys - margin)) +
            cfg.get('reweight_plpd', 0.0) * (1.0 / torch.exp(-1.0 * plpd))
        )
        entropys = entropys * coeff
    loss = entropys.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return logits, ema, False

@torch.enable_grad()
def forward_and_adapt_batchhedgev6(x, model, optimizer, margin, reset_constant_em, ema, cfg, beta: float, patch_size: int):
    logits = model(x)
    optimizer.zero_grad()
    entropys = softmax_entropy(logits)
    if cfg.get('filter_ent', True):
        filter_ids_1 = torch.where(entropys < cfg.get('deyo_margin', margin))[0]
    else:
        filter_ids_1 = torch.where(entropys <= math.log(cfg.get('num_classes', 1000)))[0]
    if len(filter_ids_1) == 0:
        return logits, ema, False
    x_sel = x[filter_ids_1].detach().clone()
    logits_sel = logits[filter_ids_1]
    # gain negative augmentation images
    x_bh6 = batch_hedge_v6_images(x_sel, patch_size=patch_size)
    with torch.no_grad():
        logits_prime = model(x_bh6)
    # gain average bias offset feature
    shuffle_logits = logits_prime.mean(dim=0, keepdim=True).expand(logits_sel.size(0), -1)
    hedge_logits = logits_sel - beta * shuffle_logits
    # predict images using debiased feature
    prob = hedge_logits.softmax(1)
    prob_prime = shuffle_logits.softmax(1)
    cls1 = prob.argmax(dim=1)
    plpd = torch.gather(prob, 1, cls1.view(-1, 1)) - torch.gather(prob_prime, 1, cls1.view(-1, 1))
    plpd = plpd.view(-1)
    if cfg.get('filter_plpd', True):
        filter_ids_2 = torch.where(plpd > cfg.get('plpd_threshold', 0.0))[0]
    else:
        filter_ids_2 = torch.where(plpd >= -2.0)[0]
    if len(filter_ids_2) == 0:
        return logits, ema, False
    entropys = softmax_entropy(hedge_logits)[filter_ids_2]
    plpd = plpd[filter_ids_2]
    if cfg.get('reweight_ent', 0.0) > 0 or cfg.get('reweight_plpd', 0.0) > 0:
        coeff = (
            cfg.get('reweight_ent', 0.0) * (1.0 / torch.exp(entropys - margin)) +
            cfg.get('reweight_plpd', 0.0) * (1.0 / torch.exp(-1.0 * plpd))
        )
        entropys = entropys * coeff
    loss = entropys.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return logits, ema, False

class DeYO(TTABase):
    def __init__(self, clip_model, class_names, cfg):
        super(DeYO, self).__init__()
        self.cfg = cfg
        clip_model.eval()
        self.clip_model = clip_model
        configure_model(self.clip_model, freeze_text_encoder=True, freeze_image_encoder=False)
        template = "a photo of a {}."
        with torch.no_grad():
            self.clip_weights = encode_text_single(clip_model, class_names, template)
        self.uni_model = UnimodalCLIP(clip_model, self.clip_weights)
        self.num_classes, self.feat_dim = self.clip_weights.shape
        self.grad_dim = sum(p.numel() for p in self.clip_model.parameters() if p.requires_grad)
        self.optimizer = optim.Adam(self.uni_model.parameters(), lr=self.cfg['lr'])
        self.copy_model_and_optimizer()
        self.steps = cfg.get('steps', 1)
        self.margin_e0 = cfg['e_margin_multiplier'] * math.log(self.num_classes)
        self.deyo_margin = cfg['deyo_margin']
        self.reset_constant_em = cfg['reset_constant_em']
        self.ema = None
        self.style = cfg.get('style', 'default')
        self.beta = cfg.get('beta', 0.0)
        self.patch_size = cfg.get('Hedge', {}).get('patch_size', 32)
        self.batchhedgev6_patch_size = cfg.get('Panda', {}).get('patch_size', self.patch_size)

    def forward(self, images):
        for _ in range(self.steps):
            if self.style.lower() == 'default':
                # ETA without Panda
                outputs, ema, reset_flag = forward_and_adapt_default(
                    images,
                    self.uni_model,
                    self.optimizer,
                    self.margin_e0,
                    self.reset_constant_em,
                    self.ema,
                    self.cfg
                )
            elif self.style == 'Panda':
                outputs, ema, reset_flag = forward_and_adapt_batchhedgev6(
                    images,
                    self.uni_model,
                    self.optimizer,
                    self.margin_e0,
                    self.reset_constant_em,
                    self.ema,
                    self.cfg,
                    self.beta,
                    self.batchhedgev6_patch_size
                )
            else:
                raise ValueError(f"Unsupported style: {self.style}")
            if reset_flag:
                self.reset()
            self.ema = ema
        return outputs

    def reset(self):
        self.load_model_and_optimizer()
        self.ema = None