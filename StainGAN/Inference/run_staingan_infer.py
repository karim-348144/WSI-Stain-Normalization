#!/usr/bin/env python3
import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

from models.models import create_model


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def discover_images(slide_dir: Path, recursive: bool):
    if recursive:
        return sorted([p for p in slide_dir.rglob("*") if p.is_file() and is_img(p)])
    return sorted([p for p in slide_dir.iterdir() if p.is_file() and is_img(p)])


def safe_slide_id(path: Path) -> str:
    try:
        return path.resolve().name
    except Exception:
        return path.name


def tensor_to_pil(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float()
        if x.ndim == 4:
            x = x[0]
        x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)
        arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        return Image.fromarray(arr)

    import numpy as np

    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 4:
            arr = arr[0]

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if arr.min() < 0.0:
                arr = (arr + 1.0) / 2.0
            elif arr.max() > 1.0:
                arr = arr / 255.0
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).round().astype("uint8")

        return Image.fromarray(arr)

    raise TypeError(f"Unsupported type for tensor_to_pil: {type(x)}")


def load_manifest(manifest_csv):
    if manifest_csv is None:
        return None, {}

    manifest_path = Path(manifest_csv)
    if not manifest_path.is_file():
        raise SystemExit(f"ERROR: --manifest_csv not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if "slide_id" not in df.columns:
        raise SystemExit(f"ERROR: manifest missing required column 'slide_id': {manifest_path}")

    tss_by_slide = {}
    if "tss" in df.columns:
        tss_by_slide = dict(zip(df["slide_id"].astype(str), df["tss"].astype(str)))

    return df, tss_by_slide


def build_test_opt(args, in_root: Path):
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()

    opt = SimpleNamespace()
    opt.isTrain = False
    opt.dataroot = str(in_root)
    opt.name = args.name
    opt.checkpoints_dir = args.checkpoints_dir
    opt.which_epoch = str(args.epoch)

    opt.model = "test"
    opt.dataset_mode = "single"
    opt.which_direction = "AtoB"

    opt.input_nc = 3
    opt.output_nc = 3
    opt.ngf = 64
    opt.ndf = 64
    opt.which_model_netG = "resnet_9blocks"
    opt.which_model_netD = "basic"
    opt.n_layers_D = 3
    opt.norm = "instance"
    opt.init_type = "normal"
    opt.no_dropout = True

    opt.loadSize = args.load_size
    opt.fineSize = args.fine_size
    opt.resize_or_crop = "none"
    opt.no_flip = True
    opt.batchSize = 1
    opt.serial_batches = True
    opt.nThreads = 1
    opt.max_dataset_size = float("inf")

    opt.display_id = 0
    opt.display_port = 8097
    opt.display_winsize = 256
    opt.results_dir = "./results"
    opt.aspect_ratio = 1.0
    opt.phase = "test"
    opt.how_many = float("inf")
    opt.gpu_ids = [0] if use_cuda else []

    return opt


def save_jpeg(img: Image.Image, out_p: Path, jpg_quality: int):
    out_p.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_p, format="JPEG", quality=jpg_quality)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="Root containing raw slide patch folders")
    ap.add_argument("--out_root", required=True, help="Root to write normalized slide folders")
    ap.add_argument("--checkpoints_dir", required=True, help="Parent checkpoints dir")
    ap.add_argument("--name", required=True, help="Experiment name, e.g. stainGAN_full")
    ap.add_argument("--epoch", default="67", help="Checkpoint epoch to load")
    ap.add_argument("--manifest_csv", default=None, help="Optional CSV with slide_id and optionally tss")
    ap.add_argument("--target_tss", default="BH", help="Slides with this TSS are copied unchanged")
    ap.add_argument("--slide_id", default=None, help="Optional single slide_id")
    ap.add_argument("--recursive", action="store_true", help="Recursively read images within slide dir")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--max_slides", type=int, default=None, help="Process only first N slides")
    ap.add_argument("--max_patches", type=int, default=None, help="Process only first M patches per slide")
    ap.add_argument("--load_size", type=int, default=256)
    ap.add_argument("--fine_size", type=int, default=256)
    ap.add_argument("--jpg_quality", type=int, default=95)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    if not in_root.is_dir():
        raise SystemExit(f"ERROR: --in_root not found or not a dir: {in_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    manifest_df, tss_by_slide = load_manifest(args.manifest_csv)

    slide_dirs = sorted([p for p in in_root.iterdir() if p.is_dir()])

    if manifest_df is not None:
        keep = set(manifest_df["slide_id"].astype(str))
        slide_dirs = [d for d in slide_dirs if safe_slide_id(d) in keep]
        if not args.quiet:
            print(f"Filtered to {len(slide_dirs)} slides from manifest ({len(manifest_df)} rows)")

    if args.slide_id is not None:
        slide_dirs = [d for d in slide_dirs if safe_slide_id(d) == args.slide_id]
        if len(slide_dirs) == 0:
            raise SystemExit(f"ERROR: requested --slide_id not found after filtering: {args.slide_id}")

    if args.max_slides is not None:
        slide_dirs = slide_dirs[:max(0, int(args.max_slides))]

    opt = build_test_opt(args, in_root)
    model = create_model(opt)
    if hasattr(model, "eval"):
        model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.fine_size, args.fine_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if not args.quiet:
        print(f"IN_ROOT={in_root}")
        print(f"OUT_ROOT={out_root}")
        print(f"SLIDES_SELECTED={len(slide_dirs)}")
        print(f"TARGET_TSS={args.target_tss}")
        print(f"MAX_PATCHES_PER_SLIDE={args.max_patches}")
        print(f"RECURSIVE={args.recursive} OVERWRITE={args.overwrite}")
        print(f"EPOCH={args.epoch} EXP_NAME={args.name}")
        print(f"DEVICE={args.device} CUDA_AVAILABLE={torch.cuda.is_available()}")

    for sd in slide_dirs:
        slide = safe_slide_id(sd)
        slide_tss = tss_by_slide.get(slide, None)

        out_sd = out_root / slide
        out_sd.mkdir(parents=True, exist_ok=True)

        files = discover_images(sd, recursive=args.recursive)
        if args.max_patches is not None:
            files = files[:max(0, int(args.max_patches))]

        mode = "copy" if slide_tss == args.target_tss else "translate"
        if not args.quiet:
            print(f"SLIDE={slide} TSS={slide_tss} MODE={mode} PATCHES={len(files)} OUT={out_sd}")

        for p in files:
            if args.recursive:
                rel = p.relative_to(sd)
                out_p = (out_sd / rel).with_suffix(".jpg")
            else:
                out_p = out_sd / (p.stem + ".jpg")

            if out_p.exists() and not args.overwrite:
                continue

            img = Image.open(p).convert("RGB")

            if slide_tss == args.target_tss:
                save_jpeg(img, out_p, args.jpg_quality)
                continue

            inp = transform(img).unsqueeze(0)
            if opt.gpu_ids:
                inp = inp.cuda(non_blocking=True)

            data = {"A": inp, "A_paths": [str(p)]}
            model.set_input(data)

            with torch.no_grad():
                model.test()

            visuals = model.get_current_visuals()
            if "fake_B" not in visuals:
                raise RuntimeError(f"Expected 'fake_B' in visuals, got keys={list(visuals.keys())}")

            fake_B = visuals["fake_B"]
            out_img = tensor_to_pil(fake_B)
            save_jpeg(out_img, out_p, args.jpg_quality)

    if not args.quiet:
        print("done")


if __name__ == "__main__":
    main()