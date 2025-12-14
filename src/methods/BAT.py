import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .TTABase import TTABase
from .utils import encode_text_single, configure_model

from .shuffle import batch_hedge_v6_images

class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def __call__(self, logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)

class I2TLoss(nn.Module):
    def __init__(self):
        super(I2TLoss, self).__init__()

    def __call__(self, logits, img_feats, text_norm_feats):
        labels = torch.argmax(logits.softmax(1), dim=1)
        loss = 0.0
        for l in torch.unique(labels, sorted=True).tolist():
            img_idx_embeddings = img_feats[labels == l]
            mean_feats = img_idx_embeddings.mean(0).type(text_norm_feats.dtype)
            dist = torch.matmul(mean_feats.unsqueeze(0), text_norm_feats[l].unsqueeze(0).t()).mean()
            loss += dist
        return loss / len(torch.unique(labels))

class InterMeanLoss(nn.Module):
    def __init__(self):
        super(InterMeanLoss, self).__init__()

    def __call__(self, logits, img_feats):
        labels = torch.argmax(logits.softmax(1), dim=1)
        mean_feats = []
        for l in torch.unique(labels, sorted=True).tolist():
            img_idx_embeddings = img_feats[labels == l]
            mean = img_idx_embeddings.mean(0)
            mean_feats.append(mean / mean.norm())

        cosine_sim_matrix = torch.matmul(torch.stack(mean_feats), torch.stack(mean_feats).t())
        loss = 1 - cosine_sim_matrix
        loss.fill_diagonal_(0)
        return loss.sum()

class BAT(TTABase):
    def __init__(self, clip_model, class_names, cfg):
        super(BAT, self).__init__()
        self.clip_model = clip_model
        self.class_names = class_names
        self.template = 'a photo of a {}.'

        self.style = cfg.get("style", "default")
        self.beta = cfg.get("beta", 0.6)
        self.hedge_patch_size = cfg.get("Hedge", {}).get("patch_size", 32)
        self.batchhedgev6_patch_size = cfg.get("Panda", {}).get("patch_size", self.hedge_patch_size)

        configure_model(self.clip_model,
                        freeze_text_encoder=False,
                        freeze_image_encoder=cfg['freeze_image_encoder'])

        self.softmax_entropy = Entropy()
        self.i2t_loss = I2TLoss()
        self.inter_mean_loss = InterMeanLoss()

        self.cfg = cfg
        self.optimizer = optim.AdamW(self.clip_model.parameters(), lr=self.cfg['lr'])
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        self.copy_model_and_optimizer()
        self.requires_augmentation = False

    def forward(self, images):
        self.clip_model.eval()

        if self.style == "Panda":
            # gain negative augmentation images
            aug_images = batch_hedge_v6_images(images, patch_size=self.batchhedgev6_patch_size)
        else:
            aug_images = images

        with torch.cuda.amp.autocast():
            text_pre_features = encode_text_single(
                clip_model=self.clip_model,
                class_names=self.class_names,
                template=self.template
            )
            text_features = F.normalize(text_pre_features, dim=1).half()

            image_pre_features = self.clip_model.encode_image(images)
            image_features = F.normalize(image_pre_features, dim=1).half()
            logits = 100.0 * (image_features @ text_features.T)

            if self.style == "Panda":
                aug_pre = self.clip_model.encode_image(aug_images)
                aug_feat = F.normalize(aug_pre, dim=1).half()
                mean_feat = aug_feat.mean(dim=0, keepdim=True)
                # gain average bias offset feature
                mean_feat = mean_feat.expand(image_features.size(0), -1)
                shuffle_logits = 100.0 * (mean_feat @ text_features.T)
                # predict images using debiased feature
                combined_logits = logits - self.beta * shuffle_logits
            else:
                # BAT without Panda
                combined_logits = logits

            loss = self.softmax_entropy(combined_logits.float()).mean(0)
            i2t_loss = self.i2t_loss(combined_logits.float(), image_pre_features.float(), text_features.float())
            inter_mean_loss = self.inter_mean_loss(combined_logits.float(), image_pre_features.float())
            loss -= i2t_loss
            loss -= inter_mean_loss

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            if self.style == "Panda":
                image_pre = self.clip_model.encode_image(images)
                image_feat = F.normalize(image_pre, dim=1).half()
                logits = 100.0 * (image_feat @ text_features.T)
                aug_pre = self.clip_model.encode_image(aug_images)
                aug_feat = F.normalize(aug_pre, dim=1).half()
                mean_feat = aug_feat.mean(dim=0, keepdim=True)
                mean_feat = mean_feat.expand(image_feat.size(0), -1)
                shuffle_logits = 100.0 * (mean_feat @ text_features.T)
                final_logits = logits - self.beta * shuffle_logits
            else:
                # BAT without Panda
                final_logits = logits

        return final_logits.float()

    def reset(self):
        self.load_model_and_optimizer()
