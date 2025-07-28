# ReblurKernelNet – Quick‑Start Guide

> **Why ReblurKernelNet?**
> On our latest 4 k‑pair dataset, ReblurKernelNet pushed **val LPIPS 0.052 (↓ 90 % vs. baseline 0.51)**, kept **val SSIM ≈ 0.99**, and averaged **pixel‑L1 (pix) error ≈ 0.021** after fine‑tuning – the largest perceptual‑quality gain we’ve recorded.


---

## 1 · Model Overview

* **Architecture**  Two‑scale U‑Net with **Reformer attention** and an optional **PatchGAN** discriminator. Recovers a sharp frame from 7‑frame burst inputs.
* **Loss stack**  `LPIPS + (1‑SSIM) + edge‑L1 + re‑blur consistency + (adversarial optional) + late pixel‑L1 tail` with dynamic weighting.
* **Training phases**

  1. **Pre‑train** – synthetic bursts, λ\_adv = 0, λ\_reblur ramps to 0.2.
  2. **Fine‑tune**  – real (aligned) pairs, full LPIPS every batch, λ\_reblur capped at 0.2, late pixel‑L1 (pix\_w ≈ 0.10).
* **Fixed crop size**  128 × 128 for the whole run (no progressive resize needed once λ‑schedules were tuned).

---

## 2 · Prerequisites

| Requirement | Tested                                 | Notes                                                    |
| ----------- | -------------------------------------- | -------------------------------------------------------- |
| Python      |  ≥ 3.9 (3.11 used)                     |                                                          |
| PyTorch     |  ≥ 2.2 (CUDA build)                    | 8 GB VRAM recommended                                    |
| Packages    | `torchvision lpips tqdm opencv‑python` | `pip install torch torchvision lpips tqdm opencv-python` |

---

## 3 · Dataset Layout

```
./photos
 ├─ blur   ── img0001.png
 │           img0002.png
 └─ sharp  ── img0001.png
             img0002.png
```

* **Place any dataset of blurry and matching sharp images inside the `photos/` folder.** The sub‑folders **must** be named `blur/` and `sharp/`, and filenames must be identical across the two.
* Supported extensions  `.png`, `.jpg`, `.jpeg` (case‑insensitive).
* Switching datasets later is as easy as replacing the contents of `photos/blur` and `photos/sharp`—no code changes required.

> **Choosing epoch counts**
>
> | Total image pairs | Suggested `--epochs_pre` | Suggested `--epochs_ft` |
> | ----------------- | ------------------------ | ----------------------- |
> | ≥ 2 000           | 120 (default)            | 80 (default)            |
> | 500 – 1 999       | \~60                     | \~40                    |
> | < 500             | \~30                     | \~20                    |
>
> These heuristics keep the number of parameter updates roughly proportional to dataset size, helping to avoid over‑ or under‑training.

---

## 4 · Training

### Minimal run (full 120 + 80 epoch schedule)

```bash
python reblurkernalnet.py \
  --blur_dir  ./data/blur \
  --sharp_dir ./data/sharp \
  --epochs_pre 120 \
  --epochs_ft  80  \
  --batch      2   \
  --device     cuda
```

### Typical flags

| Flag                           | Default          | Description              |
| ------------------------------ | ---------------- | ------------------------ |
| `--epochs_pre`                 |  120             | Pre‑training epochs      |
| `--epochs_ft`                  |  80              | Fine‑tuning epochs       |
| `--lr_pre` / `--lr_ft`         |  `1e‑4` / `5e‑5` | Learning‑rates           |
| `--resume_pre` / `--resume_ft` |                  | Resume checkpoints       |
| `--dry_run`                    | *off*            | Print schedule then exit |

> **Early‑stopping**   Add `--patience 10` to quit when val LPIPS stops improving for 10 epochs.

---

## 5 · Results Summary

| Metric           | Baseline (Hybrid) | ReblurKernelNet (this repo) |
| ---------------- | ----------------- | --------------------------- |
| **val LPIPS**    |  0.51             | **0.052**                   |
| **val SSIM**     |  0.98             | **0.99**                    |
| **avg pix (L1)** |  0.026            | **0.021**                   |

Numbers correspond to the checkpoint saved at fine‑tune epoch 72 (`checkpoints/reblurkernalnet_best.pt`).


---

## 6. Resuming or Finetuning

```bash
python reblurkernalnet.py \
  --blur_dir ./data/blur \
  --sharp_dir ./data/sharp \
  --resume_pre checkpoints/pretrain.pt \
  --resume_ft  checkpoints/finetune.pt
```

If only one checkpoint exists, point both args to the same file.

---

## 7. Logs & Checkpoints

* Validation **LPIPS** and **SSIM** printed at the end of every epoch.

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

## 10. Datasets

* **Text Image With Motion Blur** - https://www.kaggle.com/datasets/pbrant/text-image-with-motion-blur
* **Blur dataset** - https://www.kaggle.com/datasets/kwentar/blur-dataset
* **A Curated List of Image Deblurring Datasets** - https://www.kaggle.com/datasets/jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets