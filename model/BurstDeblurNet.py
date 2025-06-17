import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import resize, crop
from tqdm import tqdm


from blur_generator import generate_synthetic_burst

# =====================================
# DATASET: motion‑blurred ↔ sharp pairs
# =====================================
class BlurPairDataset(Dataset):
    """
    Loads motion-blurred ↔ sharp image pairs by zipping two sorted file lists.
    Each __getitem__ returns:
        burst_tensor  – shape (T, C, H, W)
        sharp_img     – shape (C, H, W)
    """
    def __init__(self,
                 blur_dir: str,
                 sharp_dir: str,
                 burst_gen,
                 img_size: tuple[int, int] = (256, 256),
                 T: int = 7,
                 crop_size: int = 224):
        self.blur_dir   = blur_dir
        self.sharp_dir  = sharp_dir
        self.img_size   = img_size
        self.T          = T
        self.crop_size  = crop_size
        self.burst_gen  = burst_gen

        exts = [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]

        # collect & sort full paths
        blur_paths  = sorted(sum([glob.glob(os.path.join(blur_dir,  f"*{e}")) for e in exts], []))
        sharp_paths = sorted(sum([glob.glob(os.path.join(sharp_dir, f"*{e}")) for e in exts], []))

        if not blur_paths or not sharp_paths:
            raise RuntimeError(f"No images found. Blur: {len(blur_paths)}, Sharp: {len(sharp_paths)}")
        if len(blur_paths) != len(sharp_paths):
            print(f"[WARN] {len(blur_paths)} blurred vs {len(sharp_paths)} sharp; pairing by index.")

        self.pairs = list(zip(blur_paths, sharp_paths))
        print(f"Loaded {len(self.pairs)} blur↔sharp pairs.")

    # -------- required torch-dataset hooks --------
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        blur_path, sharp_path = self.pairs[idx]

        # load images to float32 [0,1]
        blur  = read_image(blur_path).float() / 255.0        # (C,H,W)
        sharp = read_image(sharp_path).float() / 255.0

        # resize to common working size
        if blur.shape[1:] != self.img_size:
            blur  = resize(blur,  self.img_size)
        if sharp.shape[1:] != self.img_size:
            sharp = resize(sharp, self.img_size)

        # random same-location crop (registration-robust)
        if self.crop_size and self.crop_size < self.img_size[0]:
            i = random.randint(0, self.img_size[0] - self.crop_size)
            j = random.randint(0, self.img_size[1] - self.crop_size)
            blur  = crop(blur,  i, j, self.crop_size, self.crop_size)
            sharp = crop(sharp, i, j, self.crop_size, self.crop_size)

        # create synthetic burst from blurred frame
        burst = torch.stack(
            self.burst_gen(blur, num_variants=self.T), dim=0
        )  # (T,C,H,W)

        return burst, sharp

# =======================================
#  TRAINING with LPIPS + MS‑SSIM losses
# =======================================

# =====================
#  MODEL: BurstDeblur
# =====================
class BurstDeblurNet(nn.Module):
    """
    2-level U-Net with:
      • shared per-frame encoder (Conv → Conv → pool) ×2
      • 3-D temporal fusion + SE at the bottleneck
      • decoder: upsample → concat skip → Conv  (x2)
    Input  : (B, T, 3, H, W)  with H=W=224 (crop_size)
    Output : (B, 3, H, W)     same size as target
    """
    def __init__(self, in_ch: int = 3, base: int = 64, num_frames: int = 7):
        super().__init__()

        # ---------- encoder ----------
        def enc_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, 1, 1), nn.ReLU(inplace=True),
            )
        self.enc1 = enc_block(in_ch,  base)        # 224×224  → skip1
        self.pool1 = nn.MaxPool2d(2)               # 224 → 112
        self.enc2 = enc_block(base,  base*2)       # 112×112 → skip2
        self.pool2 = nn.MaxPool2d(2)               # 112 → 56
        self.enc3 = enc_block(base*2, base*4)      # 56×56   → bottleneck

        # ---------- temporal fusion at bottleneck ----------
        self.fuse3d = nn.Conv3d(base*4, base*4,
                                kernel_size=(num_frames, 1, 1), bias=False)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base*4, base*4 // 8, 1), nn.ReLU(inplace=True),
            nn.Conv2d(base*4 // 8, base*4, 1), nn.Sigmoid()
        )

        # ---------- decoder ----------
        def up_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, 1, 1), nn.ReLU(inplace=True),
            )
        self.up1 = nn.ConvTranspose2d(base*4, base*2, 4, 2, 1)   # 56 → 112
        self.dec1 = up_block(base*4, base*2)                      # concat skip2
        self.up2 = nn.ConvTranspose2d(base*2, base,   4, 2, 1)   # 112 → 224
        self.dec2 = up_block(base*2, base)                        # concat skip1

        self.out_conv = nn.Sequential(
            nn.Conv2d(base, in_ch, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, burst):                 # (B,T,C,H,W)
        B, T, C, H, W = burst.shape
        x = burst.view(B*T, C, H, W)

        # ----- encoder for each frame -----
        s1 = self.enc1(x)                     # (B*T, base, 224,224)
        p1 = self.pool1(s1)                   #              112,112
        s2 = self.enc2(p1)                    # (B*T,2*base,112,112)
        p2 = self.pool2(s2)                   #               56,56
        bottleneck = self.enc3(p2)            # (B*T,4*base, 56,56)

        # reshape to (B, T, C, H, W) then permute → (B, C, T, H, W)
        bt = bottleneck.view(B, T, -1, H//4, W//4).permute(0, 2, 1, 3, 4)
        fused = self.fuse3d(bt).squeeze(2)    # (B,4*base,56,56)
        fused = fused * self.se(fused)

        # reshape skips for fusion: max-reduce over time
        skip2 = s2.view(B, T, -1, H//2, W//2).max(dim=1).values   # (B,2*base,112,112)
        skip1 = s1.view(B, T, -1, H, W).max(dim=1).values         # (B,base,224,224)

        # ----- decoder -----
        u1 = self.up1(fused)                                      # 56→112
        d1 = self.dec1(torch.cat([u1, skip2], dim=1))             # concat skip2
        u2 = self.up2(d1)                                         # 112→224
        d2 = self.dec2(torch.cat([u2, skip1], dim=1))             # concat skip1
        out = self.out_conv(d2)                                   # (B,3,224,224)
        return out

def train_dataset(
    blur_dir: str,
    sharp_dir: str,
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    num_workers = 2,
    device: str | None = None,
):
    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = BlurPairDataset(blur_dir, sharp_dir, generate_synthetic_burst, img_size=(256,256), T=7, crop_size=224)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=(device.type=="cuda"))

    model = BurstDeblurNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ----- perceptual & MS‑SSIM loss -----
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        use_lpips = True
    except ImportError:
        print("[WARN] lpips not installed – skipping perceptual term")
        lpips_fn = None
        use_lpips = False
    try:
        from pytorch_msssim import ms_ssim
        use_msssim = True
    except ImportError:
        raise RuntimeError("Please pip install pytorch-msssim to use MS‑SSIM loss")

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        for burst, sharp in tqdm(loader, desc=f"Epoch {epoch}"):
            burst = burst.to(device)
            sharp = sharp.to(device)
            pred  = model(burst)



            loss = 0.0
            if use_lpips:
                loss += 0.5 * lpips_fn(pred, sharp).mean()
            if use_msssim:
                loss += 0.5 * (1 - ms_ssim(pred, sharp, data_range=1.0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch:3d}/{epochs}  Loss: {epoch_loss / len(loader):.4f}")

# ---------------
# ENTRY POINT
# ---------------
if __name__ == "__main__":
    # Update these paths to your Kaggle dataset folders
    BLUR_DIR  = "../photos/motion_blurred"   #  .. = go up from /model to /Deblurring
    SHARP_DIR = "../photos/sharp"
    train_dataset(BLUR_DIR, SHARP_DIR)
