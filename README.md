# Hybrid DeblurNet (`hybrid.py`) – Quick‑Start Guide

> **Why this script?**
> Among all the model variants we tried, the Hybrid version achieved the **lowest validation LPIPS** score, making it the recommended entry point for both training and inference.

---

## Model Overview

* **Architecture**: Reformer‑augmented U‑Net (`BurstDeblurNet`) that digests *synthetic burst stacks* and fuses them to reconstruct a sharp frame.
* **Two‑Phase Training**:

  1. **Pre‑train** – synthetic bursts only, optimizing **SSIM + edge‑L1 + sparse‑LPIPS**; GAN is *off*.
  2. **Fine‑tune** – switches to real images, turns **on full LPIPS** and gradually ramps up the **adversarial loss** (controlled by `λ_adv`).
* **Progressive Resize**: Images start at **128 × 128** for the first `--prog_epochs` and then bump to **256 × 256**.
* This recipe produced the **lowest validation LPIPS** among all variants we benchmarked.

---

## 1. Prerequisites

| Requirement       | Tested Version         | Notes                                                    |
| ----------------- | ---------------------- | -------------------------------------------------------- |
| Python            | ≥ 3.9                  | 3.11 used during development                             |
| PyTorch           | ≥ 2.2                  | CUDA build strongly recommended                          |
| CUDA toolkit      | 11.x / 12.x            | Optional but speeds up training                          |
| GPU               | ≥ 8 GB VRAM            | Lower batches if < 8 GB                                  |
| Other Python pkgs | see package list below | Includes `torchvision`, `lpips`, `tqdm`, `opencv‑python` |

> **Install everything in one go**
>
> ```bash
> pip install torch torchvision lpips tqdm opencv-python
> ```

---

## 2. Repository Setup

```bash
# 1. Clone the project
$ git clone https://github.com/<your‑org>/Deblurring.git
$ cd Deblurring

# 2. (Recommended) Create a virtual environment
$ python -m venv .venv
$ source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
$ pip install torch torchvision lpips tqdm opencv-python
```

---

## 3. Dataset Preparation

```
./data
 ├─ blur   ── img0001.png
 │           img0002.png
 │           …
 └─ sharp  ── img0001.png
             img0002.png
             …
```

* **Same file names** in `blur/` and `sharp/` are treated as corresponding pairs.
* Any image format readable by OpenCV / Pillow is fine (`.png`, `.jpg`).

> ✏️  Update the `--blur_dir` and `--sharp_dir` arguments (see below) if you use a different folder layout.

---

## 4. Training the Hybrid Model

### Minimal example

```bash
python hybrid.py \
  --blur_dir  ./data/blur \
  --sharp_dir ./data/sharp \
  --epochs_pre 60 \
  --epochs_ft  40 \
  --batch      2  \
  --device     cuda
```

### Common flags

| Flag                           | Default    | Description                                       |
| ------------------------------ | ---------- | ------------------------------------------------- |
| `--blur_dir`                   | *required* | Path to blurry input images                       |
| `--sharp_dir`                  | *required* | Path to ground‑truth sharp images                 |
| `--epochs_pre`                 | 60         | Pre‑training epochs (content + perceptual losses) |
| `--epochs_ft`                  | 40         | Fine‑tuning epochs (adds GAN + full LPIPS)        |
| `--batch`                      | 2          | Number of bursts per step; lower if you hit OOM   |
| `--lr_pre`                     | 1e‑4       | Learning rate during pre‑train                    |
| `--lr_ft`                      | 5e‑5       | Learning rate during fine‑tune                    |
| `--prog_epochs`                | 20         | Progressive‑resize boundary (start 128² → 256²)   |
| `--device`                     | `cuda`     | Use `cpu` if no GPU                               |
| `--resume_pre` / `--resume_ft` | *off*      | Paths to `.pt` checkpoints to resume from         |

> **Tip:** add `--dry_run` (if available) to print the training schedule without running.

---

## 5. Resuming or Finetuning

```bash
python hybrid.py \
  --blur_dir ./data/blur \
  --sharp_dir ./data/sharp \
  --resume_pre checkpoints/pretrain.pt \
  --resume_ft  checkpoints/finetune.pt
```

If only one checkpoint exists, point both args to the same file.

---

## 6. Logs & Checkpoints

* **Checkpoints** saved to `./checkpoints/hybrid_<timestamp>/` after each phase.
* **TensorBoard** logs at `./runs/` – launch with:

  ```bash
  tensorboard --logdir runs
  ```
* Validation **LPIPS** and **SSIM** printed at the end of every epoch.

---

## 7. Inference (Single Image or Folder)

```bash
python inference.py \
  --ckpt checkpoints/hybrid_best.pt \
  --input ./samples/blurred.png \
  --output ./results/
```

(The repo ships an `inference.py` helper; adjust path if renamed.)

---

## 8. Troubleshooting

| Symptom                                | Fix                                                                      |
| -------------------------------------- | ------------------------------------------------------------------------ |
| **`RuntimeError: CUDA out of memory`** | Lower `--batch`, set `CUDA_VISIBLE_DEVICES`, or switch to `--device cpu` |
| **`ModuleNotFoundError: 'lpips'`**     | `pip install lpips`                                                      |
| **`ValueError: image sizes …`**        | Confirm matching blur/sharp dimensions                                   |
| **Training stalls at 0 % GPU**         | Verify images are being read; check disk throughput                      |

---

## 9. Citation / Acknowledgements

This project builds upon several key works in deblurring, perceptual metrics, and efficient transformers. If you use this code or derive from these ideas, please cite the relevant papers:

* **Reformer** — Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya, *Reformer: The Efficient Transformer*, arXiv:2001.04451 (2020)
* **DeblurDiNAT** — Hanzhou Liu *et al.* *DeblurDiNAT: A Compact Model with Exceptional Generalization and Visual Fidelity on Unseen Domains*, arXiv:2403.13163 (2024)
* **Data‑Aug for SR** — Jaejun Yoo, Namhyuk Ahn, and Kyung‑Ah Sohn, *Rethinking Data Augmentation for Image Super‑Resolution: A Comprehensive Analysis and a New Strategy*, CVPR 2020
* **Pix2Pix (cGAN)** — Phillip Isola *et al.*, *Image‑to‑Image Translation with Conditional Adversarial Networks*, CVPR 2017
* **LPIPS** — Richard Zhang *et al.*, *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*, CVPR 2018
* **DeblurGAN‑v2** — Orest Kupyn *et al.*, *DeblurGAN‑v2: Deblurring (Orders‑of‑Magnitude) Faster and Better*, ICCV 2019
