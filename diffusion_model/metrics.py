import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_batch_ssim(samples, gts):
    """
    samples, gts: torch.Tensor [B, 1, H, W]
    """
    samples = samples.detach().cpu().numpy()
    gts = gts.detach().cpu().numpy()

    scores = []
    for i in range(samples.shape[0]):
        x = samples[i, 0]
        y = gts[i, 0]
        scores.append(
            ssim(x, y, data_range=y.max() - y.min())
        )

    return float(np.mean(scores))
