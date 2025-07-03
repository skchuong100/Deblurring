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
# add near your other imports
from pytorch_msssim import ssim
from torch.cuda.amp import autocast, GradScaler
import math
import time


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
        self.ref1 = nn.Identity()   # ← no attention at 256²
        self.pool1 = nn.MaxPool2d(2)

        # ---------- encoder level 2 ----------
        self.conv2 = nn.Conv2d(base, base*2, 3, 1, 1)
        self.ref2 = nn.Identity()   # ← no attention at 128²
        self.pool2 = nn.MaxPool2d(2)

        # ---------- bottleneck ----------
        self.conv3 = nn.Conv2d(base*2, base*4, 3, 1, 1)
        self.ref3 = Reformer(       # attention runs at 64² tokens
            dim         = base*4,
            depth       = 4,
            heads       = heads,
            bucket_size = bucket,   # 64 is OK for 4096 tokens
            causal      = False)

        # ---------- decoder ----------
        self.up1  = nn.ConvTranspose2d(base*4, base*2, 4, 2, 1)
        self.dec1 = nn.Conv2d(base*4, base*2, 3, 1, 1)

        self.up2  = nn.ConvTranspose2d(base*2, base, 4, 2, 1)
        self.dec2 = nn.Conv2d(base*2, base, 3, 1, 1)

        self.out  = nn.Conv2d(base, in_ch, 3, 1, 1)

    @staticmethod
    def _map2seq(x):
        # (B,C,H,W) → (B, H*W, C)
        return x.permute(0, 2, 3, 1).reshape(x.size(0), -1, x.size(1))

    @staticmethod
    def _seq2map(x, H, W):
        # (B, H*W, C) → (B,C,H,W)
        return x.reshape(x.size(0), H, W, -1).permute(0, 3, 1, 2)

    # ---------- forward ----------
    def forward(self, burst):
        B, T, C, H, W = burst.shape
        x = burst.view(B*T, C, H, W)

        e1 = F.relu(self.conv1(x))
        seq = self._map2seq(e1)          # use self._
        seq = self.ref1(seq)
        e1  = self._seq2map(seq, H, W)
        p1  = self.pool1(e1)

        e2 = F.relu(self.conv2(p1))
        seq = self._map2seq(e2)
        seq = self.ref2(seq)
        e2  = self._seq2map(seq, H//2, W//2)
        p2  = self.pool2(e2)

        b   = F.relu(self.conv3(p2))
        seq = self._map2seq(b)
        seq = self.ref3(seq)
        b   = self._seq2map(seq, H//4, W//4)

        d1  = self.up1(b)
        d1  = F.relu(self.dec1(torch.cat([d1, e2], dim=1)))
        d2  = self.up2(d1)
        d2  = F.relu(self.dec2(torch.cat([d2, e1], dim=1)))

        sharp = torch.sigmoid(self.out(d2))
        return sharp.view(B, T, C, H, W)[:, 0]

# ===========================================
#  TRAIN & SAVE (scheduler + checkpoints)
# ===========================================

# --- burst generators that ARE picklable -------------------
def burst_gen_synth(img, num_variants=7):
    return synth_burst(img, num_variants)

def burst_gen_identity(img, num_variants=1):
    return [img]            # single-frame “burst”
# -----------------------------------------------------------


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


# -------------------- tiny helper --------------------
def rand_crop(t: torch.Tensor, top, left, h, w):
    return t[..., top:top + h, left:left + w]
# -----------------------------------------------------

def train_dataset(
        blur_dir, sharp_dir,
        epochs_pre=60, epochs_ft=40,
        batch=2,
        lr_pre=1e-4, lr_ft=5e-5,
        prog_epochs=20,            # 1-20 → 128², then 256²
        device=None):

    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    # -------- backbone + GAN --------
    model = BurstDeblurNet(base=64, heads=4, bucket=64).to(device)
    D     = PatchDiscriminator().to(device)
    d_opt = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce   = nn.BCEWithLogitsLoss()
    λ_adv = 0.005
    adv_on = False                # switch on after LPIPS < 0.55

    import lpips; from pytorch_msssim import ms_ssim
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    start_wall = time.perf_counter()

    # --------------------------------------------------
    def run_phase(name, epochs, lr, burst_gen, T, ckpt):
        nonlocal model, adv_on
        best_lpips = float('inf')

        g_opt  = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        sched  = optim.lr_scheduler.CosineAnnealingLR(
                    g_opt, T_max=max(1, epochs // 2), eta_min=lr*0.01)
        scaler = GradScaler()

        prog_epochs = 15          # switch to 256 crop after epoch-15

        print(f"=== {name}: {epochs} epochs, lr={lr:.1e} ===")

        lr_frozen = False

        for ep in range(1, epochs+1):
            crop_sz = 128 if ep <= prog_epochs else 256

            # one-time LR boost at crop-switch
            if ep == prog_epochs + 1:
                for g in g_opt.param_groups:
                    g["lr"] = 5e-5
                print(f"🔄  LR reset to 5e-5 at epoch {ep} (crop 256)")

            # rebuild loaders ----------------------------------------------------
            full = BlurPairDataset(blur_dir, sharp_dir, burst_gen,
                                img_size=(256,256), T=T, crop_size=None)
            val_sz = 50 if len(full)>100 else int(0.1*len(full))
            tr_ds, va_ds = random_split(full, [len(full)-val_sz, val_sz],
                                        generator=torch.Generator().manual_seed(0))
            pin = device.type=='cuda'
            tr_ld = DataLoader(tr_ds, batch_size=batch, shuffle=True,
                            num_workers=2, pin_memory=pin)
            va_ld = DataLoader(va_ds, batch_size=batch, shuffle=False,
                            num_workers=2, pin_memory=pin)


    def run_phase(name, epochs, lr, burst_gen, T, ckpt):
        """
        * SSIM + edge-L1 + sparse-LPIPS during most of training
        * full-time LPIPS and GAN-weight ramp in last 10 epochs of Fine-tune
        * LR cosine anneal but frozen once it bottoms out
        """
        nonlocal model, adv_on, λ_adv          # ← added λ_adv
        best_lpips = float("inf")

        g_opt = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        sched = optim.lr_scheduler.CosineAnnealingLR(
            g_opt, T_max=max(1, epochs // 2), eta_min=lr * 0.01
        )
        min_lr_freeze = 2e-5                   # LR freeze threshold

        prog_epochs   = 15
        accum_steps   = 2
        lpips_every   = 1
        lpips_w       = 0.30                   # default perceptual weight
        λ_adv_initial = 0.002                  # initial GAN weight
        lr_frozen = False
        validate_every = 1

        print(f"=== {name}: {epochs} epochs, lr={lr:.1e} ===")

        for ep in range(1, epochs + 1):
            crop_sz = 128 if ep <= prog_epochs else 256

            # crop switch → reset LR and enable GAN
            if ep == prog_epochs + 1:
                for g in g_opt.param_groups:
                    g["lr"] = 5e-5                # put LR where you want it
                # ✱ UN-FREEZE the schedule so it can walk down again
                lr_frozen      = False            # <─ add this
                min_lr_freeze  = 1e-5             # optional: lower floor
                sched = optim.lr_scheduler.CosineAnnealingLR(
                            g_opt,                # restart cosine from here
                            T_max=max(1, epochs - ep),
                            eta_min=min_lr_freeze)
                adv_on = True                     # existing lines
                λ_adv  = λ_adv_initial
                print(f"🔄  LR reset to 5e-5 & GAN ON (λ_adv={λ_adv:.3f}) at ep {ep}")

            # ----- loader build (unchanged) -----
            full = BlurPairDataset(blur_dir, sharp_dir, burst_gen,
                                img_size=(256, 256), T=T, crop_size=None)
            val_sz = 50 if len(full) > 100 else int(0.1 * len(full))
            tr_ds, va_ds = random_split(
                full, [len(full) - val_sz, val_sz],
                generator=torch.Generator().manual_seed(0))
            pin = device.type == "cuda"
            tr_ld = DataLoader(tr_ds, batch_size=1, shuffle=True,  num_workers=2, pin_memory=pin)
            va_ld = DataLoader(va_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=pin)

            # -------------- TRAIN --------------
            model.train()
            run_loss = 0.0
            edge_w   = 0.02 if ep < 40 else 0.04
            cut_p    = 0.5 if crop_sz < 161 else 0.3

            # --- enable full-time LPIPS + ramp GAN in last 10 ep of Fine-tune ---
            if adv_on and crop_sz == 256 and λ_adv < 0.014:          # ↑ cap now 0.014
                λ_adv = round(min(0.014, λ_adv + 0.002), 4)         # +0.002 / epoch

            # full-time LPIPS + late ramp (unchanged except new cap)
            if name == "Fine-tune" and ep > epochs - 10:
                lpips_every = 1
                lpips_w     = 0.20            # leave as-is
                if adv_on and λ_adv < 0.014:
                    λ_adv = round(min(0.014, λ_adv + 0.002), 4)

            # -------- SSIM / edge blend (Option C) --------
            if name == "Fine-tune" and ep >= 26 and crop_sz == 256:
                # linear fade:  epoch 26→30  => 0.7→0.4
                ssim_w = max(0.4, 0.7 - (ep - 25) * 0.06)   # (0.7-0.4)/5 = 0.06
            else:
                ssim_w = 0.7

            g_opt.zero_grad()
            for step, (burst, sharp) in enumerate(
                    tqdm(tr_ld, desc=f"{name} {ep}/{epochs}", leave=False, ncols=80)):

                # CutBlur (batch=1)
                tgt = sharp[0]
                frames = []
                for frame in burst[0]:
                    if random.random() < cut_p:
                        frame, _ = cutblur(frame, tgt)
                    frames.append(frame)
                burst = torch.stack(frames, 0).unsqueeze(0).to(device)
                sharp = sharp.to(device)

                fake = model(burst)
                ss   = 1 - (ssim if crop_sz < 161 else ms_ssim)(fake, sharp, data_range=1.0)
                ed   = F.l1_loss(sobel(fake), sobel(sharp))
                g_loss = ssim_w * ss + (1 - ssim_w) * ed      # uses the new ssim_w

                # sparse or full LPIPS
                if (step % lpips_every) == 0:
                    lp = lpips_fn(fake, sharp).mean()
                    g_loss += lpips_w * lp

                # GAN loss if enabled
                if adv_on:
                    g_loss += λ_adv * bce(D(fake), torch.ones_like(D(fake)))

                (g_loss / accum_steps).backward()
                run_loss += g_loss.item()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(tr_ld):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    g_opt.step(); g_opt.zero_grad()

            tr_loss = run_loss / len(tr_ld)

            # -------------- VALIDATION --------------
            torch.cuda.empty_cache()
            model.eval(); v_lpips = 0.0
            with torch.no_grad():
                for b, s in tqdm(va_ld, desc=f"{name} val {ep}", leave=False, ncols=80):
                    b, s = b.to(device), s.to(device)
                    v_lpips += lpips_fn(model(b), s).sum().item()
            v_lpips /= len(va_ds)


            # LR step + freeze
            if not lr_frozen:
                sched.step()                                        # keep stepping
                curr_lr = g_opt.param_groups[0]["lr"]
                if curr_lr <= min_lr_freeze:
                    lr_frozen = True
                    # lock LR at its minimum from now on
                    for g in g_opt.param_groups:
                        g["lr"] = min_lr_freeze
                    print(f"🧊  LR frozen at {min_lr_freeze:.2e} from epoch {ep}")
            else:
                # scheduler is frozen – hold LR constant
                for g in g_opt.param_groups:
                    g["lr"] = min_lr_freeze

            print(f"{name} Ep{ep:3d}/{epochs} crop {crop_sz} | "
                f"train {tr_loss:.4f} | val LPIPS {v_lpips:.4f} | "
                f"LR {g_opt.param_groups[0]['lr']:.2e} | λ_adv {λ_adv:.3f}")

            if not math.isnan(v_lpips) and v_lpips < best_lpips:
                best_lpips = v_lpips
                torch.save(model.state_dict(), ckpt)



    # ---- phase 1: synthetic 7-frame burst ----
    # phase 1: synthetic 7-frame burst
    run_phase("Pre-train", epochs_pre, lr_pre,
            burst_gen_synth, T=7, ckpt="pretrain_best.pt")

    # phase 2: fine-tune on real images
    run_phase("Fine-tune", epochs_ft, lr_ft,
            burst_gen_identity, T=1, ckpt="finetune_best.pt")
    elapsed = int(time.perf_counter() - start_wall)
    h, m = divmod(elapsed, 3600)
    m, s = divmod(m, 60)
    print(f"🏁  Total runtime: {h:d} h {m:02d} m {s:02d} s")








# -------------------------------------------
#  MAIN ENTRY
# -------------------------------------------
if __name__ == "__main__":
    torch.cuda.empty_cache()
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    BLUR_DIR  = os.path.join(ROOT, "photos", "motion_blurred")
    SHARP_DIR = os.path.join(ROOT, "photos", "sharp")
    train_dataset(BLUR_DIR, SHARP_DIR)
