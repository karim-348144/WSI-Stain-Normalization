#!/usr/bin/env python3
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

MANIFEST = "/homes/wdkarim/manifests/brca_matching_slide_tss_manifest.csv"

RAW_ROOT = "/work/h2020deciderficarra_shared/TCGA/BRCA/images"
STAINGAN_ROOT = "/work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/norm_patches"

OUT_RAW = "/work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/color_statistics/color_stats_raw.csv"
OUT_STAINGAN = "/work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/color_statistics/color_stats_staingan.csv"

MAX_PATCHES = 256


def get_patch_files(slide_dir):
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted([p for p in Path(slide_dir).rglob("*") if p.is_file() and p.suffix.lower() in exts])


def slide_stats(slide_dir, max_patches=256):
    files = get_patch_files(slide_dir)[:max_patches]
    if len(files) == 0:
        return None

    rgb_means, rgb_stds = [], []
    lab_means, lab_stds = [], []

    for p in files:
        img = np.array(Image.open(p).convert("RGB"))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        flat_rgb = img.reshape(-1, 3)
        flat_lab = lab.reshape(-1, 3)

        rgb_means.append(flat_rgb.mean(axis=0))
        rgb_stds.append(flat_rgb.std(axis=0))
        lab_means.append(flat_lab.mean(axis=0))
        lab_stds.append(flat_lab.std(axis=0))

    rgb_means = np.stack(rgb_means).mean(axis=0)
    rgb_stds = np.stack(rgb_stds).mean(axis=0)
    lab_means = np.stack(lab_means).mean(axis=0)
    lab_stds = np.stack(lab_stds).mean(axis=0)

    return {
        "rgb_mean_r": rgb_means[0], "rgb_mean_g": rgb_means[1], "rgb_mean_b": rgb_means[2],
        "rgb_std_r":  rgb_stds[0],  "rgb_std_g":  rgb_stds[1],  "rgb_std_b":  rgb_stds[2],
        "lab_mean_l": lab_means[0], "lab_mean_a": lab_means[1], "lab_mean_b": lab_means[2],
        "lab_std_l":  lab_stds[0],  "lab_std_a":  lab_stds[1],  "lab_std_b":  lab_stds[2],
        "n_patches_used": len(files),
    }


def run(root, out_csv, stain_condition):
    df = pd.read_csv(MANIFEST)
    rows = []

    for _, row in df.iterrows():
        slide_id = str(row["slide_id"])
        slide_dir = Path(root) / slide_id
        if not slide_dir.is_dir():
            continue

        s = slide_stats(slide_dir, max_patches=MAX_PATCHES)
        if s is None:
            continue

        rows.append({
            "slide_id": slide_id,
            "tss": row["tss"],
            "stain_condition": stain_condition,
            **s
        })

    out = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with {len(out)} slides")


def main():
    run(RAW_ROOT, OUT_RAW, "raw")
    run(STAINGAN_ROOT, OUT_STAINGAN, "staingan")


if __name__ == "__main__":
    main()