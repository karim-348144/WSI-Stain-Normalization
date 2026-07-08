# add_tss_labels.py
import pandas as pd
from pathlib import Path

MANIFEST_PATH = "/work/ai4bio2025/stain_norm/manifests/brca_matching_slide_manifest.csv"

SPLIT_FILES = {
    "train": "/homes/wdkarim/splits/brca_matching_seed42/train.csv",
    "val":   "/homes/wdkarim/splits/brca_matching_seed42/val.csv",
    "test":  "/homes/wdkarim/splits/brca_matching_seed42/test.csv",
}

OUT_MANIFEST = "/homes/wdkarim/manifests/brca_matching_slide_tss_manifest.csv"
OUT_DIR = Path("/homes/wdkarim/splits/brca_matching_seed42_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_tss(slide_id: str) -> str:
    return str(slide_id).split("-")[1]

# Update manifest
manifest = pd.read_csv(MANIFEST_PATH)
manifest["tss"] = manifest["slide_id"].apply(extract_tss)

tss_counts = (
    manifest.groupby("tss")["slide_id"]
    .count()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"slide_id": "n_slides"})
)

print("TSS distribution:\n", tss_counts.to_string(index=False))
print(f"\nTotal unique TSS centers: {len(tss_counts)}")

manifest.to_csv(OUT_MANIFEST, index=False)
print(f"\nSaved manifest: {OUT_MANIFEST}")

# Update split CSVs
for split_name, path in SPLIT_FILES.items():
    df = pd.read_csv(path)
    df["tss"] = df["slide_id"].apply(extract_tss)
    out_path = OUT_DIR / f"{split_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved split: {out_path} ({len(df)} rows, {df['tss'].nunique()} TSS centers)")
