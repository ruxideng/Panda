import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


def encode_text(clip_model, class_names, templates, aggregate='average', return_text=False):
    all_texts = [[t.format(c.replace('_', ' ')) for t in templates] for c in class_names]
    text_features = []
    device = next(clip_model.parameters()).device
    for texts in all_texts:
        tokens = clip.tokenize(texts).to(device)
        emb = clip_model.encode_text(tokens)
        emb = F.normalize(emb, dim=1)
        if aggregate == 'average':
            emb = emb.mean(dim=0)
        text_features.append(emb)
    text_features = torch.stack(text_features, dim=0)
    text_features = F.normalize(text_features, dim=1)
    if return_text:
        return text_features, all_texts
    return text_features


def encode_text_single(clip_model, class_names, template):
    texts = [template.format(c.replace('_', ' ')) for c in class_names]
    device = next(clip_model.parameters()).device
    tokens = clip.tokenize(texts).to(device)
    emb = clip_model.encode_text(tokens)
    emb = F.normalize(emb, dim=1)
    return emb


def configure_model(model, freeze_text_encoder=False, freeze_image_encoder=False):
    model.eval()
    model.requires_grad_(False)

    if not freeze_text_encoder:
        for m in model.transformer.modules():
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
                m.eval()
                m.weight.requires_grad_(True)
                m.bias.requires_grad_(True)
        model.ln_final.eval()
        model.ln_final.weight.requires_grad_(True)
        model.ln_final.bias.requires_grad_(True)

    if not freeze_image_encoder:
        for m in model.visual.modules():
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
                m.eval()
                m.weight.requires_grad_(True)
                m.bias.requires_grad_(True)
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad_(True)
                m.bias.requires_grad_(True)

    return model


class UnimodalCLIP(nn.Module):
    def __init__(self, clip_model, text_feature):
        super().__init__()
        self.clip_model = clip_model
        self.text_feature = text_feature

    def forward(self, images):
        img_pre = self.clip_model.encode_image(images)
        img_f = F.normalize(img_pre, dim=1)
        return 100.0 * img_f @ self.text_feature.T

def softmax_entropy(x):
    return -(x.softmax(dim=1) * x.log_softmax(dim=1)).sum(dim=1)

def get_clip_logits(image_features, clip_weights, normalize=False, p=0.1, return_features=False):
    """
    Get clip logits and auxiliary outputs
    :param image_features: batch_size * feat_dim
    :param clip_weights: num_class * feat_dim
    :param normalize: whether to normalize
    :return:
    """

    # TODO: batch size > 1? item() only applies to batch_size = 1

    if normalize:
        image_features = F.normalize(image_features, dim=1)
        clip_weights = F.normalize(clip_weights, dim=1)

    batch_size = image_features.shape[0]

    if batch_size > 1:  # use augmix to generate multiple augmentations
        clip_logits = 100.0 * image_features @ clip_weights.T
        batch_entropy = softmax_entropy(clip_logits)
        selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_size * p)]
        image_features = F.normalize(image_features[selected_idx].mean(dim=0, keepdim=True), dim=1)

    clip_logits = 100.0 * image_features @ clip_weights.T  # HalfTensor if cuda else Float Tensor
    pred = clip_logits.argmax(dim=1).item()  # int
    proba = clip_logits.softmax(dim=1)  # HalfTensor if cuda else FloatTensor
    entropy = softmax_entropy(clip_logits).item()  # float

    if return_features:
        return clip_logits, pred, proba, entropy, image_features
    else:
        return clip_logits, pred, proba, entropy