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
# 1) add this import near the other imports
from torchvision.transforms.functional import gaussian_blur

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

# =========================
#  MODEL — 2‑level U‑Net
# =========================
class BurstDeblurNet(nn.Module):
    def __init__(self, in_ch=3, base=64, num_frames=7):
        super().__init__()
        def enc(cin, cout):
            return nn.Sequential(nn.Conv2d(cin,cout,3,1,1), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout,cout,3,1,1), nn.ReLU(inplace=True))
        self.enc1 = enc(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = enc(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = enc(base*2, base*4)
        self.fuse3d = nn.Conv3d(base*4, base*4, kernel_size=(num_frames,1,1), bias=False)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(base*4, base*4//8,1), nn.ReLU(inplace=True),
                                nn.Conv2d(base*4//8, base*4,1), nn.Sigmoid())
        def dec(cin, cout):
            return nn.Sequential(nn.Conv2d(cin,cout,3,1,1), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout,cout,3,1,1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(base*4, base*2, 4,2,1)
        self.dec1 = dec(base*4, base*2)
        self.up2 = nn.ConvTranspose2d(base*2, base, 4,2,1)
        self.dec2 = dec(base*2, base)
        self.out_conv = nn.Sequential(nn.Conv2d(base, in_ch, 3,1,1), nn.Sigmoid())
    def forward(self, burst):
        B,T,C,H,W = burst.shape
        x = burst.view(B*T, C, H, W)
        s1 = self.enc1(x); p1 = self.pool1(s1)
        s2 = self.enc2(p1); p2 = self.pool2(s2)
        bott = self.enc3(p2)
        bt = bott.view(B,T,-1,H//4,W//4).permute(0,2,1,3,4)
        fused = self.fuse3d(bt).squeeze(2); fused = fused*self.se(fused)
        skip2 = s2.view(B,T,-1,H//2,W//2).max(dim=1).values
        skip1 = s1.view(B,T,-1,H   ,W   ).max(dim=1).values
        u1 = self.up1(fused); d1 = self.dec1(torch.cat([u1,skip2],1))
        u2 = self.up2(d1);   d2 = self.dec2(torch.cat([u2,skip1],1))
        return self.out_conv(d2)

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

# ------------------------------------------------------------------
#  Replace your current train_dataset() with this version
# ------------------------------------------------------------------
def train_dataset(blur_dir, sharp_dir, epochs=100, batch=4, lr=1e-4, device=None):
    global g_sigma_max, motion_k_choices

    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    # ---- curriculum parameters (reset each run) ----
    g_sigma_max      = 7.0
    motion_k_choices = [7, 11, 15, 21]
    curriculum_unlocked = False

    # ---- model & optimiser ----
    model = BurstDeblurNet().to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)  # wd added later if needed
    cos_lr = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs // 2), eta_min=2.5e-6)  # lower floor

    # ---- data ----
    full = BlurPairDataset(blur_dir, sharp_dir, synth_burst,
                           img_size=(256, 256), T=7, crop_size=224)
    val_sz   = 50 if len(full) > 100 else int(0.1 * len(full))
    train_ds, val_ds = random_split(full, [len(full) - val_sz, val_sz],
                                    generator=torch.Generator().manual_seed(0))
    pin = device.type == 'cuda'
    train_ld = DataLoader(train_ds, batch, True, num_workers=2, pin_memory=pin)
    val_ld   = DataLoader(val_ds,  batch, False, num_workers=2, pin_memory=pin)

    # ---- losses ----
    import lpips; from pytorch_msssim import ms_ssim
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    best_lpips = float('inf')
    reg_added  = False            # tracks if over-fit response already applied

    for epoch in range(1, epochs + 1):
        # ========== TRAIN ==========
        model.train()
        run_loss = run_lpips = 0.0
        edge_w   = 0.02 if epoch < 40 else 0.04

        for burst, sharp in tqdm(train_ld, desc=f"Ep {epoch}/{epochs}"):
            burst, sharp = burst.to(device), sharp.to(device)
            pred = model(burst)

            lp   = lpips_fn(pred, sharp).mean()
            ss   = 1 - ms_ssim(pred, sharp, data_range=1.0)
            ed   = F.l1_loss(sobel(pred), sobel(sharp))
            loss = 0.45 * lp + 0.45 * ss + edge_w * ed

            opt.zero_grad(); loss.backward(); opt.step()

            run_loss  += loss.item()
            run_lpips += lp.item()

        n_batches  = len(train_ld)
        tr_loss    = run_loss  / n_batches
        tr_lpips   = run_lpips / n_batches

        # ========== VALIDATE ==========
        model.eval()
        v_lpips = 0.0
        with torch.no_grad():
            for b, s in val_ld:
                b, s = b.to(device), s.to(device)
                p    = model(b)
                v_lpips += lpips_fn(p, s).sum().item()
        v_lpips /= len(val_ds)

        # ---- GAP & over-fit response ----
        lpips_gap = tr_lpips - v_lpips        # positive gap => train better than val

        if not reg_added and lpips_gap > 0.03:
            print(f"⚠️  Over-fit trigger (LPIPS gap {lpips_gap:.3f}) – "
                  "halving LR & adding weight-decay")
            for g in opt.param_groups:
                g['lr'] *= 0.5
                g['weight_decay'] = 1e-4
            reg_added = True

        # ---- curriculum unlock ----
        if (not curriculum_unlocked) and v_lpips < 0.52:
            curriculum_unlocked = True
            print("🔓  Curriculum unlocked – enabling blur ramp & LR cosine anneal")

        if curriculum_unlocked:
            cos_lr.step()
            if epoch % 20 == 0 and g_sigma_max < 12.0:
                g_sigma_max += 0.5
                if motion_k_choices[-1] < 31:
                    motion_k_choices.append(motion_k_choices[-1] + 4)

        # ---- console ----
        lr_now = opt.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{epochs} "
              f"train {tr_loss:.4f} | "
              f"val LPIPS {v_lpips:.4f} (gap {lpips_gap:+.3f}) | "
              f"LR {lr_now:.2e}")

        # ---- checkpoints ----
        if v_lpips < best_lpips:
            best_lpips = v_lpips
            torch.save(model.state_dict(), "burst_deblur_best.pt")

    torch.save(model.state_dict(), "burst_deblur_final.pt")




# -------------------------------------------
#  MAIN ENTRY
# -------------------------------------------
if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    BLUR_DIR  = os.path.join(ROOT, "photos", "motion_blurred")
    SHARP_DIR = os.path.join(ROOT, "photos", "sharp")
    train_dataset(BLUR_DIR, SHARP_DIR)
