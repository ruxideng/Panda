import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import clip
import math

def batch_hedge_v6_images(batch, patch_size=32):
    B, C, H, W = batch.shape
    num_ph, num_pw = H // patch_size, W // patch_size
    if num_ph == 0 or num_pw == 0:
        return batch

    eff_H, eff_W = num_ph * patch_size, num_pw * patch_size

    patches = (
        batch[:, :, :eff_H, :eff_W]
        .unfold(2, patch_size, patch_size)
        .unfold(3, patch_size, patch_size)
    )
    patches = (
        patches.permute(0, 2, 3, 1, 4, 5)
        .reshape(B, num_ph * num_pw, C, patch_size, patch_size)
    )

    M = math.ceil(B / 10)
    out = torch.zeros((M, C, H, W), device=batch.device, dtype=batch.dtype)

    for m in range(M):
        base_idx = torch.randint(0, B, (1,), device=batch.device).item()

        idx_img = torch.randint(0, B, (num_ph * num_pw,), device=batch.device)
        mask = idx_img == base_idx
        while mask.any():
            idx_img[mask] = torch.randint(
                0, B, (mask.sum().item(),), device=batch.device
            )
            mask = idx_img == base_idx
        patch_idx = torch.arange(num_ph * num_pw, device=batch.device)
        new_patches = patches[idx_img, patch_idx]  # (num_patches, C, p, p)

        # Reconstruct shuffled region
        out_region = (
            new_patches
            .reshape(num_ph, num_pw, C, patch_size, patch_size)
            .permute(2, 0, 3, 1, 4)
            .reshape(C, eff_H, eff_W)
        )

        # Combine with base image
        out_batch = batch[base_idx].clone()
        out_batch[:, :eff_H, :eff_W] = out_region
        out[m] = out_batch

    return out