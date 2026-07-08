#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

MANIFEST = "/homes/wdkarim/manifests/brca_matching_slide_tss_manifest.csv"
RAW_ROOT = Path("/work/h2020deciderficarra_shared/TCGA/BRCA/images")
OUT_DIR = Path("/homes/wdkarim/stain_norm/staingan_data/BH_vs_nonBH_256")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MAX_PATCHES_A = 256

df = pd.read_csv(MANIFEST)

def get_patches(slide_dir: Path, max_p=None):
    files = sorted([p for p in slide_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if max_p is not None:
        files = files[:max_p]
    return files

lines_A, lines_B = [], []

for _, row in df.iterrows():
    slide_dir = RAW_ROOT / str(row["slide_id"])
    if not slide_dir.is_dir():
        continue

    if str(row["tss"]) == "BH":
        patches = get_patches(slide_dir, max_p=None)
        lines_B.extend(str(p) for p in patches)
    else:
        patches = get_patches(slide_dir, max_p=MAX_PATCHES_A)
        lines_A.extend(str(p) for p in patches)

(OUT_DIR / "trainA.txt").write_text("\n".join(lines_A) + "\n", encoding="utf-8")
(OUT_DIR / "trainB.txt").write_text("\n".join(lines_B) + "\n", encoding="utf-8")

print(f"trainA (non-BH, max {MAX_PATCHES_A}/slide): {len(lines_A):,} patches")
print(f"trainB (BH, all patches): {len(lines_B):,} patches")
