import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from .utils import encode_text
from .shuffle import batch_hedge_v6_images


class CLIP(nn.Module):
    def __init__(self, clip_model, class_names, cfg):
        super(CLIP, self).__init__()

        clip_model.eval()
        self.clip_model = clip_model

        templates = ['a photo of a {}.']
        with torch.no_grad():
            self.clip_weights = encode_text(clip_model, class_names, templates, aggregate='average')

        self.requires_augmentation = False

        self.style = cfg.get("style", "default")
        self.beta = cfg.get("beta", 0.6)
        self.patch_size = cfg.get("patch_size", 32)

    @torch.no_grad()
    def forward(self, images):
        if images.dim() != 4:
            raise Exception('Image must be 4D')

        img_feat = F.normalize(self.clip_model.encode_image(images), dim=1)
        orig_logits = 100.0 * (img_feat @ self.clip_weights.T)

        if self.style == "default":
            # CLIP without Panda
            return orig_logits.detach()

        elif self.style == "Panda":
            # gain negative augmentation images
            bh6 = batch_hedge_v6_images(images, patch_size=self.patch_size)
            bh6_feat = F.normalize(self.clip_model.encode_image(bh6), dim=1)
            # gain average bias offset feature
            mean_feat = bh6_feat.mean(dim=0, keepdim=True).expand(img_feat.size(0), -1)
            # predict images using debiased feature
            logits = orig_logits - self.beta * (100.0 * (mean_feat @ self.clip_weights.T))
            return logits.detach()

        else:
            raise ValueError(f"Unsupported style: {self.style}")

    def reset(self):
        pass
