import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import clip
import math

from .utils import encode_text, configure_model, encode_text_single
from .TTABase import TTABase

from .shuffle import batch_hedge_v6_images

class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()
    def __call__(self, logits):
        return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)

class Tent(TTABase):

    def __init__(self, clip_model, class_names, cfg):
        super(Tent, self).__init__()

        clip_model.eval()
        self.clip_model = clip_model

        configure_model(self.clip_model, freeze_text_encoder=True, freeze_image_encoder=False)

        with torch.no_grad():
            template = "a photo of a {}."
            self.clip_weights = encode_text_single(clip_model, class_names, template)

        self.num_classes, self.feat_dim = self.clip_weights.shape
        self.grad_dim = sum(param.numel() for param in self.clip_model.parameters() if param.requires_grad)

        self.softmax_entropy = Entropy()

        self.cfg = cfg
        self.style = cfg.get("style", "default")
        self.beta = cfg.get("beta", 0.2)
        self.hedge_patch_size = cfg.get("Hedge", {}).get("patch_size", 32)
        self.batchhedgev6_patch_size = cfg.get("Panda", {}).get("patch_size", self.hedge_patch_size)

        self.optimizer = optim.Adam(self.clip_model.parameters(), lr=self.cfg['lr'])

        self.copy_model_and_optimizer()

    def forward(self, images):
        assert images.dtype == torch.float32

        style = self.style
        with torch.cuda.amp.autocast():
            image_pre_features = self.clip_model.encode_image(images)
            image_features = F.normalize(image_pre_features, dim=1)
            orig_logits = 100.0 * image_features @ self.clip_weights.T

            if style == "default":
                # Tent without Panda
                loss = self.softmax_entropy(orig_logits).mean()
            elif style == "Panda":
                # gain negative augmentation images
                bh6 = batch_hedge_v6_images(images, patch_size=self.batchhedgev6_patch_size)
                img_feat = F.normalize(self.clip_model.encode_image(images), dim=1)
                bh6_feat = F.normalize(self.clip_model.encode_image(bh6), dim=1)
                # gain average bias offset feature
                mean_feat = bh6_feat.mean(dim=0, keepdim=True).expand(img_feat.size(0), -1)
                bh6_logits = 100.0 * mean_feat @ self.clip_weights.T
                loss = self.softmax_entropy(orig_logits - self.beta * bh6_logits).mean()
            else:
                raise ValueError(f"Unsupported style: {self.style}")

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        with torch.no_grad():
            image_pre_features = self.clip_model.encode_image(images)
            image_features = F.normalize(image_pre_features, dim=1)
            orig_logits = 100.0 * image_features @ self.clip_weights.T

            if style == "default":
                out_logits = orig_logits
            elif style == "Panda":
                # gain negative augmentation images
                bh6 = batch_hedge_v6_images(images, patch_size=self.batchhedgev6_patch_size)
                img_feat = F.normalize(self.clip_model.encode_image(images), dim=1)
                bh6_feat = F.normalize(self.clip_model.encode_image(bh6), dim=1)
                # gain average bias offset feature
                mean_feat = bh6_feat.mean(dim=0, keepdim=True).expand(img_feat.size(0), -1)
                bh6_logits = 100.0 * mean_feat @ self.clip_weights.T
                # predict images using debiased feature
                out_logits = orig_logits - self.beta * bh6_logits
            else:
                out_logits = orig_logits

        return out_logits

    def reset(self):
        self.load_model_and_optimizer()
