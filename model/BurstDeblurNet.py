import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import resize

from blur_generator import generate_synthetic_burst

# ----- Dataset -----
class DeblurDataset(Dataset):
    def __init__(self, clean_dir, burst_gen, img_size=(256,256), T=7):
        # Absolute path to clean images
        clean_dir = r"C:\Users\spenc\OneDrive\Documents\Deblurring\photos\clean"
        # Collect image paths for common extensions
        patterns = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
        paths = []
        for pat in patterns:
            paths.extend(glob.glob(os.path.join(clean_dir, pat)))
        # Remove duplicates and sort
        self.paths = sorted(set(paths))

        # Optional debugging
        print(f"Loading clean images from: {clean_dir}")
        print(f"Found {len(self.paths)} files:")
        for p in self.paths:
            print("  ", p)

        self.burst_gen = burst_gen
        self.img_size = img_size
        self.T = T

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = read_image(path).float() / 255.0  # (C,H,W)
        if img.shape[1:] != self.img_size:
            img = resize(img, self.img_size)
        # Generate burst on-the-fly
        burst = self.burst_gen(img, num_variants=self.T)
        burst_tensor = torch.stack(burst, dim=0)  # (T,C,H,W)
        return burst_tensor, img

# ----- Model -----
class BurstDeblurNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, num_frames=7):
        super().__init__()
        # shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, 1, 1), nn.ReLU(inplace=True)
        )
        # 3D fusion over time
        self.fuse3d = nn.Conv3d(base_ch, base_ch, kernel_size=(num_frames,1,1), bias=False)
        # SE attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_ch, base_ch//8, 1), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch//8, base_ch, 1), nn.Sigmoid()
        )
                # --- Simple decoder: no skip concat to avoid size mismatch ---
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch//2, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch//2, base_ch//4, 3, 1, 1), nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(base_ch//4, in_ch, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, burst):  # (B,T,C,H,W)
        B, T, C, H, W = burst.shape
        x = burst.view(B*T, C, H, W)
        feat = self.encoder(x)                       # (B*T, F, H, W)
        feat = feat.view(B, T, -1, H, W)             # (B, T, F, H, W)
        feat = feat.permute(0, 2, 1, 3, 4)           # (B, F, T, H, W)
        fused = self.fuse3d(feat).squeeze(2)         # (B, F, H, W)
        fused = fused * self.se(fused)               # SE attention

        # --- Decoder (upsample-only path) ---
        fused = fused * self.se(fused)   # still 256×256
        d = self.dec1(fused)             # 256×256
        out = self.final(d)              # 256×256
        return out

# ----- Training Loop -----
def train(
    clean_dir,
    epochs=200,
    batch_size=4,
    lr=1e-4,
    device=None
):
    # choose device automatically
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # dataset & loader
    dataset = DeblurDataset(clean_dir, generate_synthetic_burst, img_size=(256,256), T=7)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # model, optimizer, losses
    model = BurstDeblurNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    l1_loss = nn.L1Loss()

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        for burst, clean in loader:
            burst = burst.to(device)
            clean = clean.to(device)
            pred = model(burst)
            loss = 0.6 * l1_loss(pred, clean)
            # add perceptual and MS-SSIM here

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch}/{epochs}  Loss: {epoch_loss/len(loader):.4f}")

if __name__ == '__main__':
    # use absolute path or relative as needed
    clean_dir = r"C:\Users\spenc\OneDrive\Documents\Deblurring\photos\clean"
    train(clean_dir)
