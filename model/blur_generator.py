# blur_generator.py

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import gaussian_blur, rotate

def add_gaussian_noise(img_tensor, std=0.05):
    noise = torch.randn_like(img_tensor) * std
    return torch.clamp(img_tensor + noise, 0.0, 1.0)

def apply_motion_blur(img: torch.Tensor, kernel_size: int, angle: float) -> torch.Tensor:
    """
    img: (C, H, W), values ∈ [0,1]
    kernel_size: odd int, length of the motion streak
    angle: in degrees, direction of motion
    """
    # 1) create a horizontal line kernel
    kernel = torch.zeros(kernel_size, kernel_size, dtype=img.dtype, device=img.device)
    kernel[kernel_size // 2, :] = 1.0

    # 2) rotate it by `angle`
    #    rotate expects a Tensor of shape (..., H, W)
    kernel = rotate(kernel.unsqueeze(0), angle=angle).squeeze(0)
    kernel = kernel / kernel.sum()  # normalize

    # 3) convolve per-channel
    C, H, W = img.shape
    k = kernel_size // 2
    kernel = kernel.unsqueeze(0).unsqueeze(0)            # (1,1,K,K)
    kernel = kernel.expand(C, 1, kernel_size, kernel_size)
    img_b = img.unsqueeze(0)                             # (1,C,H,W)
    blurred = F.conv2d(img_b, kernel, padding=k, groups=C)
    return blurred.squeeze(0)

def generate_synthetic_burst(img_tensor: torch.Tensor,
                             num_variants: int = 5) -> list[torch.Tensor]:
    burst = []
    for _ in range(num_variants):
        # — Gaussian blur parameters —
        g_k = int(np.random.choice([9, 11, 13, 15]))            # cast to Python int
        g_sigma = float(np.random.uniform(3.0, 7.0))

        # — Motion blur parameters —
        m_k = int(np.random.choice([7, 11, 15, 21]))            # length of streak
        m_angle = float(np.random.uniform(0, 360))

        # apply gaussian
        g = gaussian_blur(img_tensor, kernel_size=g_k, sigma=g_sigma)

        # apply motion
        m = apply_motion_blur(g, kernel_size=m_k, angle=m_angle)

        # add noise
        n_std = float(np.random.uniform(0.03, 0.08))
        final = add_gaussian_noise(m, std=n_std)

        burst.append(final)

    return burst
