import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import clip
import math
from .utils import encode_text_single, configure_model, UnimodalCLIP, encode_text
from .TTABase import TTABase

from .shuffle import batch_hedge_v6_images

@torch.jit.script
def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)

@torch.enable_grad()
def forward_and_adapt_default(x, model, optimizer, fishers, e_margin, current_model_probs,
                             fisher_alpha, d_margin, num_samples_update, d_margin_val):
    outputs = model(x)
    entropys = softmax_entropy(outputs)
    filter_ids_1 = torch.where(entropys < e_margin)[0]
    if len(filter_ids_1) == 0:
        return outputs, 0, 0, current_model_probs
    x_prime = x[filter_ids_1].detach().clone()
    entropys = entropys[filter_ids_1]
    if current_model_probs is not None:
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(0),
                                                  outputs[filter_ids_1].softmax(1), dim=1)
        filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin_val)[0]
        entropys = entropys[filter_ids_2]
        updated_probs = update_model_probs(current_model_probs,
                                           outputs[filter_ids_1][filter_ids_2].softmax(1))
        num_counts_2 = filter_ids_2.size(0)
    else:
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
        num_counts_2 = entropys.size(0)
        filter_ids_2 = list(range(entropys.size(0)))
    if entropys.numel() == 0:
        return outputs, 0, filter_ids_1.size(0), updated_probs
    coeff = 1 / torch.exp(entropys.detach() - e_margin)
    entropys = entropys * coeff
    loss = entropys.mean()
    if fishers is not None:
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fishers:
                ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1]) ** 2).sum()
        loss += ewc_loss
    if x_prime[filter_ids_2].size(0) != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    return outputs, num_counts_2, filter_ids_1.size(0), updated_probs

@torch.enable_grad()
def forward_and_adapt_batchhedgev6(x, model, optimizer, fishers, e_margin, current_model_probs,
                                  fisher_alpha, d_margin, num_samples_update, beta, patch_size):
    outputs_clean = model(x)
    # gain negative augmentation images
    x_bh6 = batch_hedge_v6_images(x, patch_size)
    outputs_bh6 = model(x_bh6)
    # gain average bias offset feature
    mean_logits = outputs_bh6.mean(dim=0, keepdim=True).expand(outputs_clean.size(0), -1)
    # predict images using debiased feature
    hedge_outputs = outputs_clean - beta * mean_logits

    entropys = softmax_entropy(hedge_outputs)
    filter_ids = torch.where(entropys < e_margin)[0]
    if len(filter_ids) == 0:
        return hedge_outputs, 0, 0, current_model_probs

    entropys_sel = entropys[filter_ids]
    updated_probs = update_model_probs(current_model_probs,
        hedge_outputs[filter_ids].softmax(1))

    coeff = 1 / torch.exp(entropys_sel.detach() - e_margin)
    loss = (entropys_sel * coeff).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return hedge_outputs, entropys_sel.numel(), filter_ids.numel(), updated_probs

class ETA(TTABase):
    def __init__(self, clip_model, class_names, cfg):
        super(ETA, self).__init__()
        clip_model.eval()
        self.clip_model = clip_model
        configure_model(self.clip_model, freeze_text_encoder=True, freeze_image_encoder=False)
        template = "a photo of a {}."
        with torch.no_grad():
            self.clip_weights = encode_text_single(clip_model, class_names, template)
        self.uni_model = UnimodalCLIP(clip_model, self.clip_weights)
        self.num_classes, self.feat_dim = self.clip_weights.shape
        self.grad_dim = sum(p.numel() for p in self.clip_model.parameters() if p.requires_grad)
        self.cfg = cfg
        self.optimizer = optim.Adam(self.uni_model.parameters(), lr=cfg["lr"])
        self.copy_model_and_optimizer()
        self.steps = cfg.get("steps", 1)
        self.style = cfg.get("style", "default")
        self.beta = cfg.get("beta", 0.0)
        self.batchhedgev6_patch_size = cfg.get("Panda", {}).get("patch_size", 32)
        self.e_margin = cfg["e_margin_multiplier"] * math.log(self.num_classes)
        self.d_margin = cfg["d_margin"]
        self.fishers = None
        self.fisher_alpha = 2000.0
        self.current_model_probs = None

    def forward(self, images):
        for _ in range(self.steps):
            style = self.style
            if style == "default":
                # ETA without Panda
                outputs, cnt2, cnt1, updated = forward_and_adapt_default(
                    images, self.uni_model, self.optimizer, self.fishers,
                    self.e_margin, self.current_model_probs,
                    self.fisher_alpha, self.d_margin,
                    self.num_samples_update_2 if hasattr(self, "num_samples_update_2") else 0,
                    self.d_margin
                )
            elif style == "Panda":
                outputs, cnt2, cnt1, updated = forward_and_adapt_batchhedgev6(
                    images, self.uni_model, self.optimizer, self.fishers,
                    self.e_margin, self.current_model_probs,
                    self.fisher_alpha, self.d_margin,
                    getattr(self, "num_samples_update_2", 0),
                    self.beta, self.batchhedgev6_patch_size
                )
            else:
                raise ValueError(f"Unsupported style: {self.style}")
            self.num_samples_update_2 = getattr(self, "num_samples_update_2", 0) + cnt2
            self.num_samples_update_1 = getattr(self, "num_samples_update_1", 0) + cnt1
            self.current_model_probs = updated
        return outputs

    def reset(self):
        self.load_model_and_optimizer()
        self.current_model_probs = None
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0