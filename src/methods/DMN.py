import torch
import torch.nn as nn
import torch.nn.functional as F
from bisect import bisect_left
from .TTABase import TTABase
from .utils import encode_text_single, encode_text

from .shuffle import batch_hedge_v6_images

def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(dim=1)
    return -(p * p.log()).sum(dim=1)

def get_clip_logits(image_features: torch.Tensor, clip_weights: torch.Tensor, normalize: bool = False, p: float = 0.1, return_features: bool = False):
    if normalize:
        image_features = F.normalize(image_features, dim=1)
        clip_weights = F.normalize(clip_weights, dim=1)
    B = image_features.shape[0]
    if B > 1:
        logits_all = 100.0 * (image_features @ clip_weights.T)
        ent_all = softmax_entropy(logits_all)
        k = max(1, int(B * (1 - p)))
        idx = torch.argsort(ent_all, descending=False)[:k]
        image_features = F.normalize(image_features[idx].mean(dim=0, keepdim=True), dim=1)
    clip_logits = 100.0 * (image_features @ clip_weights.T)
    pred = clip_logits.argmax(dim=1)
    proba = clip_logits.softmax(dim=1)
    ent = softmax_entropy(clip_logits)
    if return_features:
        return clip_logits, pred, proba, ent, image_features
    return clip_logits, pred, proba, ent

class DMN(TTABase):
    def __init__(self, clip_model, class_names, cfg):
        super(DMN, self).__init__()
        clip_model.eval()
        self.clip_model = clip_model
        self.device = next(self.clip_model.parameters()).device
        template = "a photo of a {}."
        with torch.no_grad():
            self.clip_weights = encode_text_single(clip_model, class_names, template)
        self.num_class, self.feat_dim = self.clip_weights.shape
        self.style = cfg.get("style", "default")
        self.beta = float(cfg.get("beta", 0.0))
        self.hedge_patch_size = int(cfg.get("Hedge", {}).get("patch_size", 32))
        self.batchhedgev6_patch_size = int(cfg.get("Panda", {}).get("patch_size", self.hedge_patch_size))
        pos_cfg = cfg.get("positive", {})
        self.pos_enabled = bool(pos_cfg.get("enabled", False))
        if self.pos_enabled:
            self.shot_capacity = int(pos_cfg.get("shot_capacity", 30))
            self.pos_alpha = float(pos_cfg.get("alpha", 0.6))
            self.pos_beta = float(pos_cfg.get("beta", 15.0))
        else:
            self.shot_capacity = 0
            self.pos_alpha = 0.0
            self.pos_beta = 1.0
        if self.pos_enabled:
            self.pos_cache = torch.zeros((self.num_class, self.shot_capacity, self.feat_dim), device=self.device, dtype=self.clip_weights.dtype)
            self.pos_cache_cnt = torch.zeros(self.num_class, device=self.device, dtype=torch.long)
            self.pos_ent = [[] for _ in range(self.num_class)]
        else:
            self.pos_cache = None
            self.pos_cache_cnt = None
            self.pos_ent = None

    @torch.no_grad()
    def update_cache(self, pred_label: int, feature: torch.Tensor, entropy: float):
        c = pred_label
        ent_list = self.pos_ent[c]
        if len(ent_list) >= self.shot_capacity:
            old_ent, old_idx = ent_list[-1]
            if entropy < old_ent:
                self.pos_cache[c, old_idx] = feature
                ent_list.pop()
                new_item = (entropy, old_idx)
                insert_idx = bisect_left(ent_list, new_item)
                ent_list.insert(insert_idx, new_item)
        else:
            idx = len(ent_list)
            self.pos_cache_cnt[c] += 1
            self.pos_cache[c, idx] = feature
            new_item = (entropy, idx)
            insert_idx = bisect_left(ent_list, new_item)
            ent_list.insert(insert_idx, new_item)

    @torch.no_grad()
    def compute_cache_proba(self, image_feature: torch.Tensor) -> torch.Tensor:
        C, K, D = self.pos_cache.shape
        affinity = (self.pos_cache @ image_feature.T).squeeze(-1)
        attn = torch.exp(self.pos_beta * (affinity - 1.0))
        weighted = (self.pos_cache * attn.unsqueeze(-1)).sum(dim=1)
        adaptive_cls_weight = F.normalize(weighted, dim=1)
        zero_mask = (self.pos_cache_cnt == 0)
        adaptive_cls_weight[zero_mask] = 0.0
        cache_logits = 100.0 * (image_feature @ adaptive_cls_weight.T)
        cache_proba = cache_logits.softmax(dim=1)
        return self.pos_alpha * cache_proba

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        with torch.no_grad():
            raw_feats = self.clip_model.encode_image(images)
            img_feats = F.normalize(raw_feats, dim=1)
        if self.style == "Panda":
            # gain negative augmentation images
            bh6 = batch_hedge_v6_images(images, patch_size=self.batchhedgev6_patch_size)
            bh6_feats = F.normalize(self.clip_model.encode_image(bh6), dim=1)  # (M, D)
            # gain average bias offset feature
            mean_feat = bh6_feats.mean(dim=0, keepdim=True)  # (1, D)
            mean_feat = mean_feat.expand(img_feats.size(0), -1)  # (B, D)
            # predict images using debiased feature
            shuffle_logits_all = 100.0 * (mean_feat @ self.clip_weights.T)
        final_logits_list = []
        for i in range(B):
            img = images[i: i + 1]
            feat = img_feats[i: i + 1]
            orig_logits = 100.0 * (feat @ self.clip_weights.T)
            sl = self.style
            if sl == "Panda":
                aug_logits = shuffle_logits_all[i: i + 1]
                combined_logits = orig_logits - self.beta * aug_logits
            else:
                # DMN without Panda
                combined_logits = orig_logits
            proba = combined_logits.softmax(dim=1)
            ent = softmax_entropy(combined_logits)
            pred_label = int(combined_logits.argmax(dim=1).item())
            if self.pos_enabled:
                self.update_cache(pred_label, feat, float(ent.item()))
            cache_proba = self.compute_cache_proba(feat) if self.pos_enabled else torch.zeros_like(proba)
            final_proba = proba + cache_proba
            final_proba = final_proba / final_proba.sum(dim=1, keepdim=True)
            final_logits = torch.log(final_proba + 1e-12)
            final_logits_list.append(final_logits)
        return torch.cat(final_logits_list, dim=0)

    def reset(self):
        if self.pos_enabled:
            self.pos_cache.zero_()
            self.pos_cache_cnt.zero_()
            for ent_list in self.pos_ent:
                ent_list.clear()
