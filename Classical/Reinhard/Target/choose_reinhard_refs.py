#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def list_slides(in_root: Path):
    return sorted([p for p in in_root.iterdir() if p.is_dir()])


def list_patches(slide_dir: Path, recursive: bool):
    it = slide_dir.rglob("*") if recursive else slide_dir.iterdir()
    files = [p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files)


def imread_rgb(p: Path):
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {p}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def rgb_to_lab_stats(img_rgb: np.ndarray):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    flat = lab.reshape(-1, 3)
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0) + 1e-6
    return mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", default="/work/h2020deciderficarra_shared/TCGA/BRCA/images/",
                    help="Root containing slide subfolders (patch images inside).")
    ap.add_argument("--n_slides", type=int, default=100,
                    help="Use first N slides (sorted) for computing slide statistics.")
    ap.add_argument("--n_patches", type=int, default=1024,
                    help="Use first M patches per slide (sorted) for computing slide statistics.")
    ap.add_argument("--k_refs", type=int, default=20,
                    help="Number of reference slides selected closest to the global median.")
    ap.add_argument("--recursive", action="store_true",
                    help="Recursively look for patch images inside each slide folder.")
    ap.add_argument("--out_json", default="/work/ai4bio2025/stain_norm/targets/reinhard_template_agg.json",
                    help="Output JSON containing aggregate template (mu_lab/sd_lab) and selected ref slides.")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_json = Path(args.out_json)

    if not in_root.is_dir():
        raise SystemExit(f"ERROR: --in_root not found or not a dir: {in_root}")

    slides = list_slides(in_root)[: args.n_slides]
    if len(slides) == 0:
        raise SystemExit(f"ERROR: No slide subfolders found in {in_root}")

    per_slide = []
    for sd in slides:
        patches = list_patches(sd, recursive=args.recursive)[: args.n_patches]
        if len(patches) == 0:
            continue

        mus, sds = [], []
        for p in patches:
            img = imread_rgb(p)
            mu, sdv = rgb_to_lab_stats(img)
            mus.append(mu)
            sds.append(sdv)

        mu_slide = np.median(np.stack(mus), axis=0)
        sd_slide = np.median(np.stack(sds), axis=0)
        vec = np.concatenate([mu_slide, sd_slide])

        per_slide.append({
            "slide": sd.name,
            "mu_lab": mu_slide.tolist(),
            "sd_lab": sd_slide.tolist(),
            "vec": vec.tolist(),
            "n_patches_used": len(patches),
        })

    if len(per_slide) == 0:
        raise SystemExit("ERROR: No patches found in the selected slides (check --recursive and folder structure).")

    V = np.stack([x["vec"] for x in per_slide])
    med = np.median(V, axis=0)
    d = np.linalg.norm(V - med[None, :], axis=1)

    order = np.argsort(d)
    k = min(args.k_refs, len(order))
    refs = [per_slide[i] for i in order[:k]]

    mu_agg = np.median(np.stack([r["mu_lab"] for r in refs]), axis=0)
    sd_agg = np.median(np.stack([r["sd_lab"] for r in refs]), axis=0)

    payload = {
        "in_root": str(in_root),
        "n_slides_requested": int(args.n_slides),
        "n_slides_used": int(len(per_slide)),
        "n_patches_per_slide": int(args.n_patches),
        "k_refs_requested": int(args.k_refs),
        "k_refs_used": int(k),
        "recursive": bool(args.recursive),
        "ref_slides": [r["slide"] for r in refs],
        "mu_lab": mu_agg.tolist(),
        "sd_lab": sd_agg.tolist(),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    print("Wrote:", out_json)
    print("Refs:", ", ".join(payload["ref_slides"]))


if __name__ == "__main__":
    main()

