import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from .TTABase import TTABase
from .utils import encode_text_single, configure_model
from .shuffle import batch_hedge_v6_images

def select_confident_samples(logits, top=0.1):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    n_keep = max(1, int(batch_entropy.size(0) * top))
    idx = torch.argsort(batch_entropy, descending=False)[:n_keep]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

class TPS(TTABase):
    def __init__(self, clip_model, class_names, cfg):
        super(TPS, self).__init__()
        self.device = next(clip_model.parameters()).device
        clip_model.eval()
        self.clip_model = clip_model
        self.config = cfg
        configure_model(self.clip_model, freeze_text_encoder=True, freeze_image_encoder=False)
        self.clip_weights = encode_text_single(clip_model, class_names, "a photo of a {}.")
        D = self.clip_weights.shape[1]
        K = self.clip_weights.shape[0]
        self.shift_vectors = nn.Parameter(torch.zeros(K, D, device=self.device))
        self.optimizer = optim.AdamW([self.shift_vectors], lr=cfg["lr"])
        self.copy_model_and_optimizer()
        self.optimizer_initial_state = deepcopy(self.optimizer.state_dict())
        self.style = cfg.get("style", "default")
        self.beta = cfg.get("beta", 0.0)
        self.hedge_patch_size = cfg.get("patch_size", 32)

    def forward(self, images, feats=None):
        if feats is not None and self.config.get("use_cache", True):
            B = feats.shape[0]
            V = feats.shape[1]
        else:
            B = images.shape[0]
            V = 1
        logits_list = []

        # Prepare mean_feat if needed
        mean_feat = None
        if self.style == "Panda":
            # gain negative augmentation images
            bh6 = batch_hedge_v6_images(images, patch_size=self.hedge_patch_size)
            with torch.no_grad():
                # gain average bias offset feature
                bh6_feat = F.normalize(self.clip_model.encode_image(bh6), dim=1)
            mean_feat = bh6_feat.mean(dim=0, keepdim=True)  # (1, D)

        for b in range(B):
            self.shift_vectors.data.zero_()
            self.optimizer.load_state_dict(deepcopy(self.optimizer_initial_state))

            if feats is not None and self.config.get("use_cache", True):
                img_feat_all = feats[b].clone()  # [V, D]
            else:
                img_feat_all = F.normalize(self.clip_model.encode_image(images[b:b+1]), dim=1)  # [1, D]

            if mean_feat is not None:
                img_feat_all = img_feat_all - self.beta * mean_feat

            for _ in range(self.config.get("update_steps", 1)):
                proto = (self.clip_weights + self.shift_vectors).to(img_feat_all.dtype)
                shifted = F.normalize(proto, dim=-1)
                logits_batch = 100.0 * img_feat_all @ shifted.T  # [V, K]
                output, idx = select_confident_samples(logits_batch, self.config.get("selection_p", 0.1))
                loss = avg_entropy(output)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                proto = (self.clip_weights + self.shift_vectors).to(img_feat_all.dtype)
                shifted = F.normalize(proto, dim=-1)
                if feats is not None and self.config.get("use_cache", True):
                    orig_feat = feats[b, 0].clone()
                else:
                    orig_feat = F.normalize(self.clip_model.encode_image(images[b:b+1]), dim=1).squeeze(0)
                if mean_feat is not None:
                    # predict images using debiased feature
                    orig_feat = orig_feat - self.beta * mean_feat.squeeze(0)
                logits_b = 100.0 * orig_feat @ shifted.T
            logits_list.append(logits_b)

        return torch.stack(logits_list, dim=0)

    def reset(self):
        self.load_model_and_optimizer()
