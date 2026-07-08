#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os

from huggingface_hub import login

if "HF_TOKEN" in os.environ and os.environ["HF_TOKEN"].strip():
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
XY_RE = re.compile(r"x_(\d+)_y_(\d+)", re.IGNORECASE)


def is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def safe_slide_id(path: Path) -> str:
    try:
        return path.resolve().name
    except Exception:
        return path.name


def discover_images(slide_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in slide_dir.rglob("*") if p.is_file() and is_img(p)]
    else:
        files = [p for p in slide_dir.iterdir() if p.is_file() and is_img(p)]
    return sorted(files)


def parse_xy_from_name(name: str) -> Tuple[int, int]:
    m = XY_RE.search(name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return -1, -1


def select_slide_dirs(in_root: Path, manifest_csv: str = None, slide_id: str = None, max_slides: int = None):
    slide_dirs = sorted([p for p in in_root.iterdir() if p.is_dir()])

    if manifest_csv:
        manifest_path = Path(manifest_csv)
        if not manifest_path.is_file():
            raise SystemExit(f"ERROR: --manifest_csv not found: {manifest_path}")
        df = pd.read_csv(manifest_path)
        if "slide_id" not in df.columns:
            raise SystemExit(f"ERROR: manifest missing required column 'slide_id': {manifest_path}")
        keep = set(df["slide_id"].astype(str))
        slide_dirs = [d for d in slide_dirs if safe_slide_id(d) in keep]

    if slide_id is not None:
        slide_dirs = [d for d in slide_dirs if safe_slide_id(d) == slide_id]
        if len(slide_dirs) == 0:
            raise SystemExit(f"ERROR: requested --slide_id not found after filtering: {slide_id}")

    if max_slides is not None:
        slide_dirs = slide_dirs[: max(0, int(max_slides))]

    return slide_dirs


class PatchDataset(Dataset):
    def __init__(self, files: List[Path], transform):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        x, y = parse_xy_from_name(p.stem)
        return self.transform(img), str(p), x, y


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    paths = [b[1] for b in batch]
    xs = torch.tensor([b[2] for b in batch], dtype=torch.int64)
    ys = torch.tensor([b[3] for b in batch], dtype=torch.int64)
    return imgs, paths, xs, ys


def load_uni_model(device: torch.device):
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
        num_classes=0,
    )
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval().to(device)
    return model, transform


def main():
    ap = argparse.ArgumentParser(description="Extract UNI features from raw patch folders")
    ap.add_argument("--in_root", required=True, help="Root containing raw patch slide folders")
    ap.add_argument("--out_root", required=True, help="Output root for per-slide feature .pt files")
    ap.add_argument("--manifest_csv", default=None, help="Optional CSV with slide_id column")
    ap.add_argument("--slide_id", default=None, help="Optional single slide_id to process")
    ap.add_argument("--recursive", action="store_true", help="Recursively find patches inside each slide folder")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output .pt files")
    ap.add_argument("--quiet", action="store_true", help="Reduce logging")
    ap.add_argument("--max_slides", type=int, default=None, help="Only first N slides after filtering")
    ap.add_argument("--max_patches", type=int, default=None, help="Only first M patches per slide after sorting")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    if not in_root.is_dir():
        raise SystemExit(f"ERROR: --in_root not found or not a dir: {in_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA requested but not available")

    model, transform = load_uni_model(device)

    slide_dirs = select_slide_dirs(
        in_root=in_root,
        manifest_csv=args.manifest_csv,
        slide_id=args.slide_id,
        max_slides=args.max_slides,
    )

    if not args.quiet:
        print(f"IN_ROOT={in_root}")
        print(f"OUT_ROOT={out_root}")
        print(f"SLIDES_SELECTED={len(slide_dirs)}")
        print(f"DEVICE={device}")
        print(f"BATCH_SIZE={args.batch_size}")
        print(f"NUM_WORKERS={args.num_workers}")

    for sd in slide_dirs:
        slide = safe_slide_id(sd)
        files = discover_images(sd, recursive=args.recursive)

        if args.max_patches is not None:
            files = files[: max(0, int(args.max_patches))]

        out_pt = out_root / f"{slide}.pt"

        if out_pt.exists() and not args.overwrite:
            if not args.quiet:
                print(f"SKIP slide={slide} existing={out_pt}")
            continue

        if len(files) == 0:
            if not args.quiet:
                print(f"SKIP slide={slide} no image files")
            continue

        if not args.quiet:
            print(f"SLIDE={slide} PATCHES={len(files)} OUT={out_pt}")

        ds = PatchDataset(files, transform)
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
        )

        feat_chunks = []
        path_list = []
        x_chunks = []
        y_chunks = []

        with torch.inference_mode():
            for imgs, paths, xs, ys in tqdm(dl, desc=f"uni:{slide}", disable=args.quiet, leave=False):
                imgs = imgs.to(device, non_blocking=True)
                feats = model(imgs)
                feat_chunks.append(feats.cpu())
                path_list.extend(paths)
                x_chunks.append(xs.cpu())
                y_chunks.append(ys.cpu())

        features = torch.cat(feat_chunks, dim=0)
        xs = torch.cat(x_chunks, dim=0)
        ys = torch.cat(y_chunks, dim=0)

        payload = {
            "slide_id": slide,
            "features": features,
            "x": xs,
            "y": ys,
            "patch_paths": path_list,
            "model_name": "MahmoodLab/UNI",
            "feature_dim": int(features.shape[1]),
            "num_patches": int(features.shape[0]),
            "source_root": str(in_root),
        }
        torch.save(payload, out_pt)

        if not args.quiet:
            print(f"WROTE={out_pt} SHAPE={tuple(features.shape)}")

    if not args.quiet:
        print("done")


if __name__ == "__main__":
    main()
