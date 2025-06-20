# ------------------------------------------------------------
# Burst Deblur Training Script (updated)
# ------------------------------------------------------------
# Changes in this version
#   • Adds StepLR scheduler (halve LR every 20 epochs)
#   • Saves best checkpoint to "burst_deblur_best.pt"
#   • Saves final weights to user‑specified path (default "burst_deblur_final.pt")
# ------------------------------------------------------------

import os, glob, random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import resize, crop
from tqdm import tqdm
from reformer_pytorch import Reformer
# 1) add this import near the other imports
from torchvision.transforms.functional import gaussian_blur
import random
import numpy as np
from blur_generator import generate_synthetic_burst, add_gaussian_noise, apply_motion_blur


g_sigma_max = 7.0
motion_k_choices = [7, 11, 15, 21]
# ===============================
#  DATASET — Blur ↔ Sharp pairs
# ===============================
class BlurPairDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, burst_gen, img_size=(256,256), T=7, crop_size=224):
        self.img_size, self.T, self.crop_size, self.burst_gen = img_size, T, crop_size, burst_gen
        exts = [".png",".PNG",".jpg",".JPG",".jpeg",".JPEG"]
        blur = sorted(sum([glob.glob(os.path.join(blur_dir,f"*{e}")) for e in exts],[]))
        sharp= sorted(sum([glob.glob(os.path.join(sharp_dir,f"*{e}")) for e in exts],[]))
        if len(blur)!=len(sharp): print("[WARN] count mismatch; pairing by index")
        self.pairs=list(zip(blur,sharp))
        print(f"Loaded {len(self.pairs)} blur↔sharp pairs")
    def __len__(self): return len(self.pairs)
    def __getitem__(self,idx):
        bp,sp=self.pairs[idx]
        b=read_image(bp).float()/255.; s=read_image(sp).float()/255.
        if b.shape[1:]!=self.img_size: b=resize(b,self.img_size)
        if s.shape[1:]!=self.img_size: s=resize(s,self.img_size)
        if self.crop_size and self.crop_size<self.img_size[0]:
            i=random.randint(0,self.img_size[0]-self.crop_size)
            j=random.randint(0,self.img_size[1]-self.crop_size)
            b=crop(b,i,j,self.crop_size,self.crop_size)
            s=crop(s,i,j,self.crop_size,self.crop_size)
        burst=torch.stack(self.burst_gen(b,num_variants=self.T),0)
        return burst,s

from reformer_pytorch import Reformer

class BurstDeblurNet(nn.Module):
    """
    Two-scale U-Net with Reformer attention:
      • Captures long-range blur streaks without O(N²) cost
      • Keeps parameter count close to original (~7 M with base=64)
      • Accepts a burst (B,T,C,H,W) and returns one sharp frame
    """
    def __init__(self, in_ch=3, base=64, heads=4, bucket=64, num_frames=7):
        super().__init__()
        # ---------- encoder level 1 ----------
        self.conv1 = nn.Conv2d(in_ch, base, 3, 1, 1)
        self.ref1  = Reformer(
            dim         = base,
            depth       = 2,
            heads       = heads,
            bucket_size = bucket,
            causal      = False)
        self.pool1 = nn.MaxPool2d(2)

        # ---------- encoder level 2 ----------
        self.conv2 = nn.Conv2d(base, base*2, 3, 1, 1)
        self.ref2  = Reformer(
            dim         = base*2,
            depth       = 2,
            heads       = heads,
            bucket_size = bucket,
            causal      = False)
        self.pool2 = nn.MaxPool2d(2)

        # ---------- bottleneck ----------
        self.conv3 = nn.Conv2d(base*2, base*4, 3, 1, 1)
        self.ref3  = Reformer(
            dim         = base*4,
            depth       = 4,
            heads       = heads,
            bucket_size = bucket,
            causal      = False)

        # ---------- decoder ----------
        self.up1  = nn.ConvTranspose2d(base*4, base*2, 4, 2, 1)
        self.dec1 = nn.Conv2d(base*4, base*2, 3, 1, 1)

        self.up2  = nn.ConvTranspose2d(base*2, base, 4, 2, 1)
        self.dec2 = nn.Conv2d(base*2, base, 3, 1, 1)

        self.out  = nn.Conv2d(base, in_ch, 3, 1, 1)

    def forward(self, burst):
        B, T, C, H, W = burst.shape
        x = burst.view(B*T, C, H, W)          # (B·T, C, H, W)

        e1 = F.relu(self.conv1(x))
        e1 = self.ref1(e1)
        p1 = self.pool1(e1)

        e2 = F.relu(self.conv2(p1))
        e2 = self.ref2(e2)
        p2 = self.pool2(e2)

        b  = F.relu(self.conv3(p2))
        b  = self.ref3(b)

        d1 = self.up1(b)
        d1 = F.relu(self.dec1(torch.cat([d1, e2], dim=1)))

        d2 = self.up2(d1)
        d2 = F.relu(self.dec2(torch.cat([d2, e1], dim=1)))

        sharp = torch.sigmoid(self.out(d2))
        return sharp.view(B, T, C, H, W)[:, 0]    # return the first (fused) frame


# ===========================================
#  TRAIN & SAVE (scheduler + checkpoints)
# ===========================================


def sobel(img: torch.Tensor) -> torch.Tensor:
    """Compute per-pixel gradient magnitude (Sobel) for a 3-channel image tensor
       img: (B, 3, H, W) in [0,1].
       returns: (B, 3, H, W) gradient mag per channel.
    """
    kernel_x = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    kernel_y = torch.tensor([[-1,-2,-1],
                             [ 0, 0, 0],
                             [ 1, 2, 1]], dtype=img.dtype, device=img.device).view(1,1,3,3)

    # depthwise conv for each channel
    gx = F.conv2d(img, kernel_x.repeat(img.size(1),1,1,1), padding=1, groups=img.size(1))
    gy = F.conv2d(img, kernel_y.repeat(img.size(1),1,1,1), padding=1, groups=img.size(1))
    return torch.sqrt(gx*gx + gy*gy + 1e-6)

def synth_burst(img: torch.Tensor, num_variants: int = 7):
    burst = []
    for _ in range(num_variants):
        g_k     = int(np.random.choice([9, 11, 13, 15]))        # kernel size (odd)
        g_sigma = float(np.random.uniform(3.0, g_sigma_max))    # σ for gaussian blur

        m_k     = int(np.random.choice(motion_k_choices))       # motion-blur length
        m_angle = float(np.random.uniform(0, 360))              # motion-blur angle

        n_std   = float(np.random.uniform(0.03, 0.08))          # additive-noise σ

        # ---- blur & noise pipeline ----
        g_blur  = gaussian_blur(img, kernel_size=g_k, sigma=g_sigma)        # Gaussian blur
        m_blur  = apply_motion_blur(g_blur, kernel_size=m_k, angle=m_angle) # Motion blur
        final   = add_gaussian_noise(m_blur, std=n_std)                      # Add noise

        burst.append(final)
    return burst


def cutblur(img_blur, img_sharp, alpha=0.7):
    """
    Half the time, mix a sharp patch into blur (or vice-versa).
    img_blur, img_sharp: (C,H,W) in [0,1]
    Returns two tensors: mixed_input, mixed_target.
    """
    if random.random() > 0.5:
        return img_blur, img_sharp          # no change
    _, h, w = img_blur.shape
    cut_ratio = random.uniform(0.3, alpha)
    ch, cw   = int(h * cut_ratio), int(w * cut_ratio)
    cy, cx   = random.randint(0, h - ch), random.randint(0, w - cw)
    mixed_in  = img_blur.clone()
    mixed_tg  = img_sharp.clone()
    if random.random() < 0.5:   # paste SHARP patch into BLUR
        mixed_in[:, cy:cy+ch, cx:cx+cw] = img_sharp[:, cy:cy+ch, cx:cx+cw]
    else:                       # paste BLUR patch into SHARP
        mixed_tg[:, cy:cy+ch, cx:cx+cw] = img_blur[:, cy:cy+ch, cx:cx+cw]
    return mixed_in, mixed_tg

# ---------------- PatchGAN discriminator ----------------
class PatchDiscriminator(nn.Module):
    """3-layer 70×70 PatchGAN (very light)."""
    def __init__(self, ch_in=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in,   base,     4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base,    base*2,   4, 2, 1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base*2,  base*4,   4, 2, 1), nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base*4,  1,        3, 1, 1)   # logits
        )
    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
#  Replace your current train_dataset() with this version
# ------------------------------------------------------------------
def train_dataset(
        blur_dir, sharp_dir,
        epochs_pre=60, epochs_ft=40,
        batch=4,
        lr_pre=1e-4, lr_ft=5e-5,
        prog_epochs=20,                 # epochs with 128×128 crops
        device=None):

    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    # -------- backbone --------
    model = BurstDeblurNet(base=64, heads=4, bucket=64).to(device)

    # -------- discriminator --------
    D        = PatchDiscriminator().to(device)
    d_opt    = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce_loss = nn.BCEWithLogitsLoss()
    λ_adv    = 0.005
    adv_on   = False                 # toggled when LPIPS < 0.55

    # -------- common perceptual losses --------
    import lpips; from pytorch_msssim import ms_ssim
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # --------------------------------------------------------
    def run_phase(name, epochs, lr, burst_gen, ckpt_path):
        nonlocal model, adv_on
        best_lpips = float('inf')

        # fresh optimiser for each phase
        g_opt  = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        sch_lr = optim.lr_scheduler.CosineAnnealingLR(
                     g_opt, T_max=max(1, epochs // 2), eta_min=lr * 0.01)

        print(f"=== {name}: {epochs} epochs, lr={lr:.1e} ===")

        for ep in range(1, epochs + 1):
            # -------- dataset / dataloader with progressive crop --------
            crop_sz = 128 if ep <= prog_epochs else 256
            full = BlurPairDataset(
                blur_dir, sharp_dir, burst_gen,
                img_size=(256, 256), T=len(burst_gen(None)),
                crop_size=None)                # we crop manually below
            val_sz = 50 if len(full) > 100 else int(0.1 * len(full))
            tr_ds, va_ds = random_split(
                full, [len(full) - val_sz, val_sz],
                generator=torch.Generator().manual_seed(0))
            pin = device.type == 'cuda'
            tr_ld = DataLoader(tr_ds, batch, True,  num_workers=2, pin_memory=pin)
            va_ld = DataLoader(va_ds, batch, False, num_workers=2, pin_memory=pin)

            # ---------------- train ----------------
            model.train()
            run_loss = run_lpips = 0.0
            edge_w   = 0.02 if ep < 40 else 0.04

            for burst, sharp in tqdm(tr_ld, desc=f"{name} Ep{ep}/{epochs}"):
                # manual random crop (progressive)
                _, H, W = sharp.shape
                if H > crop_sz:
                    top  = random.randint(0, H - crop_sz)
                    left = random.randint(0, W - crop_sz)
                    sharp = crop(sharp, top, left, crop_sz, crop_sz)
                    burst = crop(burst, top, left, crop_sz, crop_sz)

                # CutBlur / MixDegrade on each frame
                mixed_frames = []
                for f in burst:
                    m_in, m_tg = cutblur(f, sharp)
                    mixed_frames.append(m_in)
                    sharp = m_tg                     # use last mixed as target
                burst = torch.stack(mixed_frames, 0)

                burst, sharp = burst.to(device), sharp.to(device)
                fake = model(burst)

                # --- generator loss ---
                lp = lpips_fn(fake, sharp).mean()
                ss = 1 - ms_ssim(fake, sharp, data_range=1.0)
                ed = F.l1_loss(sobel(fake), sobel(sharp))
                g_loss = 0.45*lp + 0.45*ss + edge_w*ed

                if adv_on:
                    logits_fake = D(fake)
                    g_loss += λ_adv * bce_loss(logits_fake, torch.ones_like(logits_fake))

                g_opt.zero_grad(); g_loss.backward(); g_opt.step()

                # --- discriminator loss ---
                if adv_on:
                    logits_real = D(sharp)
                    logits_fake = D(fake.detach())
                    d_loss = 0.5*( bce_loss(logits_real, torch.ones_like(logits_real)) +
                                   bce_loss(logits_fake, torch.zeros_like(logits_fake)) )
                    d_opt.zero_grad(); d_loss.backward(); d_opt.step()

                run_loss  += g_loss.item()
                run_lpips += lp.item()

            tr_loss  = run_loss  / len(tr_ld)
            tr_lpips = run_lpips / len(tr_ld)

            # ---------------- validate ----------------
            model.eval(); v_lpips = 0.0
            with torch.no_grad():
                for b, s in va_ld:
                    b, s = b.to(device), s.to(device)
                    v_lpips += lpips_fn(model(b), s).sum().item()
            v_lpips /= len(va_ds)

            if (not adv_on) and v_lpips < 0.55:
                adv_on = True
                print("🎯  Adversarial loss activated")

            sch_lr.step()
            print(f"{name} Ep{ep:3d}/{epochs} "
                  f"crop {crop_sz} | train {tr_loss:.4f} "
                  f"| val LPIPS {v_lpips:.4f} | LR {g_opt.param_groups[0]['lr']:.2e}")

            if v_lpips < best_lpips:
                best_lpips = v_lpips
                torch.save(model.state_dict(), ckpt_path)

    # ---------- PHASE 1 : synthetic burst ----------
    run_phase("Pre-train", epochs_pre, lr_pre,
              lambda img, num_variants=7: synth_burst(img, num_variants),
              "pretrain_best.pt")

    # ---------- PHASE 2 : real images ----------
    run_phase("Fine-tune", epochs_ft, lr_ft,
              lambda img, num_variants=1: [img],
              "finetune_best.pt")





# -------------------------------------------
#  MAIN ENTRY
# -------------------------------------------
if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    BLUR_DIR  = os.path.join(ROOT, "photos", "motion_blurred")
    SHARP_DIR = os.path.join(ROOT, "photos", "sharp")
    train_dataset(BLUR_DIR, SHARP_DIR)
