#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from wsi_normalizer import imread, MacenkoNormalizer, ReinhardNormalizer, VahadaneNormalizer

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def get_normalizer(method: str):
    m = method.lower()
    if m == "macenko":
        return MacenkoNormalizer()
    if m == "reinhard":
        return ReinhardNormalizer()
    if m == "vahadane":
        return VahadaneNormalizer()
    raise ValueError(f"Unsupported method: {method}")


def is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def discover_images(slide_dir: Path, recursive: bool):
    if recursive:
        for p in slide_dir.rglob("*"):
            if p.is_file() and is_img(p):
                yield p
    else:
        for p in slide_dir.iterdir():
            if p.is_file() and is_img(p):
                yield p


def safe_slide_id(path: Path) -> str:
    try:
        return path.resolve().name
    except Exception:
        return path.name


def load_reinhard_template_json_for_normalizer(p: Path):
    tpl = json.loads(p.read_text())
    mu = np.array(tpl["mu_lab"], dtype=np.float32)
    sd = np.array(tpl["sd_lab"], dtype=np.float32)

    # Convert raw OpenCV LAB stats to the internal Reinhard space expected by
    # StainTools-like implementations:
    #   L channel: divide by 2.55
    #   a,b channels: subtract 128 for means
    #   stds: L scaled by 2.55, a/b unchanged
    mu_r = np.array([
        mu[0] / 2.55,
        mu[1] - 128.0,
        mu[2] - 128.0,
    ], dtype=np.float32)

    sd_r = np.array([
        sd[0] / 2.55,
        sd[1],
        sd[2],
    ], dtype=np.float32)

    mu_tuple = (
        np.array([[mu_r[0]]], dtype=np.float32),
        np.array([[mu_r[1]]], dtype=np.float32),
        np.array([[mu_r[2]]], dtype=np.float32),
    )
    sd_tuple = (
        np.array([[sd_r[0]]], dtype=np.float32),
        np.array([[sd_r[1]]], dtype=np.float32),
        np.array([[sd_r[2]]], dtype=np.float32),
    )
    return mu_tuple, sd_tuple, mu_r, sd_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="Root containing slide subfolders (patch images inside)")
    ap.add_argument("--out_root", required=True, help="Output root (slide subfolders created)")
    ap.add_argument("--method", required=True, choices=["reinhard", "macenko", "vahadane"])

    ap.add_argument("--target_img", default=None, help="Target image path for fitting (single reference)")
    ap.add_argument(
        "--reinhard_template_json",
        default=None,
        help="JSON with keys mu_lab and sd_lab (aggregate Reinhard template). If set and method=reinhard, it is used instead of --target_img.",
    )

    ap.add_argument("--jpg_quality", type=int, default=95)
    ap.add_argument("--recursive", action="store_true", help="Recursively read images within slide dir")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--quiet", action="store_true", help="Reduce logging")

    ap.add_argument("--max_slides", type=int, default=None, help="Normalize only the first N slide subfolders (sorted).")
    ap.add_argument("--max_patches", type=int, default=None, help="Normalize only the first M patch images per slide (sorted).")
    ap.add_argument("--manifest_csv", type=str, default=None, help="CSV with slide_id column. Only slides present in this manifest will be processed.")
    ap.add_argument("--slide_id", type=str, default=None, help="If set, process only this single slide_id after manifest filtering.")

    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    if not in_root.is_dir():
        raise SystemExit(f"ERROR: --in_root not found or not a dir: {in_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    normalizer = get_normalizer(args.method)

    if args.method == "reinhard" and args.reinhard_template_json:
        tpl_path = Path(args.reinhard_template_json)
        if not tpl_path.is_file():
            raise SystemExit(f"ERROR: --reinhard_template_json not found: {tpl_path}")

        mu_tuple, sd_tuple, mu_r, sd_r = load_reinhard_template_json_for_normalizer(tpl_path)
        normalizer.target_means = mu_tuple
        normalizer.target_stds = sd_tuple

        if not args.quiet:
            print(f"METHOD=reinhard TEMPLATE_JSON={tpl_path}")
            print(f"TEMPLATE_REINHARD_MEANS={mu_r.tolist()}")
            print(f"TEMPLATE_REINHARD_STDS={sd_r.tolist()}")

    else:
        if args.target_img is None:
            raise SystemExit("ERROR: Provide --target_img (or --reinhard_template_json for reinhard).")
        target_img = Path(args.target_img)
        if not target_img.is_file():
            raise SystemExit(f"ERROR: --target_img not found: {target_img}")
        normalizer.fit(imread(str(target_img)))

        if not args.quiet:
            print(f"METHOD={args.method} TARGET_IMG={target_img}")

    slide_dirs = sorted([p for p in in_root.iterdir() if p.is_dir()])

    if args.manifest_csv:
        manifest_path = Path(args.manifest_csv)
        if not manifest_path.is_file():
            raise SystemExit(f"ERROR: --manifest_csv not found: {manifest_path}")
        df = pd.read_csv(manifest_path)
        if "slide_id" not in df.columns:
            raise SystemExit(f"ERROR: manifest missing required column 'slide_id': {manifest_path}")
        slide_ids = set(df["slide_id"].astype(str))
        slide_dirs = [d for d in slide_dirs if safe_slide_id(d) in slide_ids]
        if not args.quiet:
            print(f"Filtered to {len(slide_dirs)} slides from manifest ({len(df)} total rows)")

    if args.slide_id is not None:
        slide_dirs = [d for d in slide_dirs if safe_slide_id(d) == args.slide_id]
        if len(slide_dirs) == 0:
            raise SystemExit(f"ERROR: requested --slide_id not found after filtering: {args.slide_id}")

    if args.max_slides is not None:
        slide_dirs = slide_dirs[: max(0, int(args.max_slides))]

    if not args.quiet:
        print(f"IN_ROOT={in_root}")
        print(f"OUT_ROOT={out_root}")
        print(f"SLIDES_SELECTED={len(slide_dirs)}")
        print(f"MAX_PATCHES_PER_SLIDE={args.max_patches}")
        print(f"RECURSIVE={args.recursive} OVERWRITE={args.overwrite}")

    for sd in slide_dirs:
        slide_id = safe_slide_id(sd)
        out_sd = out_root / slide_id
        out_sd.mkdir(parents=True, exist_ok=True)

        files = sorted(discover_images(sd, recursive=args.recursive))
        if args.max_patches is not None:
            files = files[: max(0, int(args.max_patches))]

        if not args.quiet:
            print(f"SLIDE={slide_id} IN_FILES_SELECTED={len(files)} OUT_DIR={out_sd}")

        for p in tqdm(files, desc=f"{args.method}:{slide_id}", leave=False, disable=args.quiet):
            if args.recursive:
                rel = p.relative_to(sd)
                out_p = (out_sd / rel).with_suffix(".jpg")
                out_p.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_p = out_sd / (p.stem + ".jpg")

            if out_p.exists() and not args.overwrite:
                continue

            img = imread(str(p))
            out = normalizer.transform(img)

            out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(
                str(out_p),
                out_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)],
            )
            if not ok:
                raise RuntimeError(f"Failed to write: {out_p}")

    if not args.quiet:
        print("done")


if __name__ == "__main__":
    main()
