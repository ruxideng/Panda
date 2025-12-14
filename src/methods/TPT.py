import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from .TTABase import TTABase
from .PromptLearner import PromptLearner, TextEncoder
from .utils import configure_model
from .shuffle import batch_hedge_v6_images

def select_confident_samples(logits, top=0.1):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    n_keep = max(1, int(logits.size(0) * top))
    idx = torch.argsort(batch_entropy, descending=False)[:n_keep]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

class CoOpCLIP(nn.Module):
    def __init__(self, clip_model, class_names, n_ctx, ctx_init, csc, device):
        super().__init__()
        self.prompt_learner = PromptLearner(clip_model, class_names, n_ctx, ctx_init, csc, device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        for name, param in self.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

    def forward(self, image_features):
        prompts = self.prompt_learner().to(torch.float32)
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        image_features = F.normalize(image_features.to(torch.float32), dim=1)
        text_features = F.normalize(text_features.to(torch.float32), dim=1)
        return 100.0 * image_features @ text_features.T

    @torch.no_grad()
    def reset(self):
        self.prompt_learner.reset()

class TPT(TTABase):
    def __init__(self, clip_model, class_names, cfg):
        super().__init__()
        clip_model = clip_model.float()
        self.device = next(clip_model.parameters()).device
        clip_model.eval()
        self.clip_model = clip_model
        self.config = cfg
        configure_model(self.clip_model, freeze_text_encoder=True, freeze_image_encoder=True)
        self.coop_model = CoOpCLIP(
            clip_model,
            class_names,
            n_ctx=cfg["n_ctx"],
            ctx_init=cfg["ctx_init"],
            csc=cfg["csc"],
            device=self.device
        ).to(self.device).float().eval()
        self.optimizer = optim.AdamW(self.coop_model.prompt_learner.parameters(), lr=cfg["lr"])
        self.opt_initial_state = deepcopy(self.optimizer.state_dict())
        self.copy_model_and_optimizer()
        self.style = cfg.get("style", "default")
        self.beta = cfg.get("beta", 0.0)
        self.hedge_patch_size = cfg.get("patch_size", 32)

    def forward(self, images, feats=None):
        B = images.size(0)
        if feats is not None and self.config.get("use_cache", True):
            feats_all = feats.to(self.device)  # [B, 64, D]
        else:
            raise ValueError("Please provide feats: tensor of shape [B,64,D]")

        # --- mean_feat preparation ---
        mean_feat = None
        if self.style == "Panda":
            # gain negative augmentation images
            bh6 = batch_hedge_v6_images(images, patch_size=self.hedge_patch_size)
            with torch.no_grad():
                # gain average bias offset feature
                bh6_feat = F.normalize(self.clip_model.encode_image(bh6), dim=1)
            mean_feat = bh6_feat.mean(dim=0, keepdim=True)  # [1, D]

        results = []
        for i in range(B):
            sample_feats = feats_all[i].clone()  # [64, D]
            if mean_feat is not None:
                # predict images using debiased feature
                sample_feats = sample_feats - self.beta * mean_feat  # broadcast

            self.coop_model.reset()
            self.coop_model.eval()
            self.optimizer.zero_grad()
            self.optimizer.load_state_dict(deepcopy(self.opt_initial_state))

            logits_all = self.coop_model(sample_feats)  # [64, K]
            selected, _ = select_confident_samples(logits_all, self.config["selection_p"])
            loss = avg_entropy(selected)

            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                logits_fin = self.coop_model(sample_feats)
            results.append(logits_fin[0].unsqueeze(0))
        return torch.cat(results, dim=0)

    def reset(self):
        self.load_model_and_optimizer()
