import torch
import torch.nn.functional as F
import math
from .TTABase import TTABase
from .utils import encode_text_single, configure_model

from .shuffle import batch_hedge_v6_images

class Zero(TTABase):
    def __init__(self, clip_model, class_names, cfg):
        super(Zero, self).__init__()
        self.device = next(clip_model.parameters()).device
        clip_model.eval()
        self.clip_model = clip_model
        self.config = cfg
        configure_model(self.clip_model, freeze_text_encoder=True, freeze_image_encoder=False)

        self.clip_weights = encode_text_single(
            clip_model, class_names, "a photo of a {}."
        ).to(self.device)  # [K, D]

        self.p = cfg.get("selection_p", 0.1)
        self.style = cfg.get("style", "default")
        self.beta = cfg.get("beta", 0.4)
        self.hedge_patch_size = cfg.get("patch_size", 32)

    def forward(self, images, feats=None):
        if feats is not None and self.config.get("use_cache", True):
            all_feats = feats.to(self.device)           # [B, V, D]
        else:
            img_feat = self.clip_model.encode_image(images.to(self.device))
            all_feats = img_feat.unsqueeze(1)           # [B, 1, D]

        B, V, D = all_feats.shape
        K = self.clip_weights.size(0)
        zero_temp = torch.finfo(all_feats.dtype).eps     # very small number

        if self.style == "Panda":
            # images: [B, C, H, W], get negative feature
            # gain negative augmentation images
            bh6 = batch_hedge_v6_images(images, patch_size=self.hedge_patch_size)  # [M, C, H, W]
            with torch.no_grad():
                # gain average bias offset feature
                bh6_feat = F.normalize(self.clip_model.encode_image(bh6), dim=1)   # [M, D]
            mean_feat = bh6_feat.mean(dim=0, keepdim=True)                        # [1, D]
        else:
            mean_feat = None  # no augmentation

        outputs = []
        for i in range(B):
            fi = all_feats[i]                         # [V, D]
            fi = F.normalize(fi, dim=1)

            # Negative Augmentation
            if mean_feat is not None:
                # predict images using debiased feature
                fi = fi - self.beta * mean_feat       # [V, D]

            logits_i = 100.0 * (fi @ self.clip_weights.T)  # [V, K]
            ent = -(logits_i.softmax(1) * logits_i.log_softmax(1)).sum(1)
            n_sel = max(1, int(ent.size(0) * self.p))
            idx = torch.argsort(ent, descending=False)[:n_sel]
            l_filt = logits_i[idx]                    # [n_sel, K]
            p_bar = (l_filt / zero_temp).softmax(1).sum(0)  # [K]
            outputs.append(p_bar)

        return torch.stack(outputs, dim=0)            # [B, K]

    def reset(self):
        pass
