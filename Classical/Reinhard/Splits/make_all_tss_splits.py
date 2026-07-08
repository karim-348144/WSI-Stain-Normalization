import json
import pandas as pd
from pathlib import Path

MANIFEST = "/homes/wdkarim/manifests/brca_matching_slide_tss_manifest.csv"
OUT_ROOT = Path("/homes/wdkarim/splits")

HOLDOUTS = [
    ("A2", "BH"),
    ("A2", "E2"),
    ("BH", "A2"),
    ("BH", "E2"),
    ("E2", "A2"),
    ("E2", "BH"),
]

keep_cols = ["slide_id", "patient_id", "time_os_months", "event_os", "tss"]

df = pd.read_csv(MANIFEST)

for test_tss, val_tss in HOLDOUTS:
    out_dir = OUT_ROOT / f"brca_tss_holdout_{test_tss}_{val_tss}"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = df[df["tss"] == test_tss].copy()
    val_df = df[df["tss"] == val_tss].copy()
    train_df = df[(df["tss"] != test_tss) & (df["tss"] != val_tss)].copy()

    train_df[keep_cols].to_csv(out_dir / "train.csv", index=False)
    val_df[keep_cols].to_csv(out_dir / "val.csv", index=False)
    test_df[keep_cols].to_csv(out_dir / "test.csv", index=False)

    meta = {
        "split_type": "unseen_center_holdout",
        "test_tss": test_tss,
        "val_tss": val_tss,
        "train_tss_count": int(train_df["tss"].nunique()),
        "val_tss_count": int(val_df["tss"].nunique()),
        "test_tss_count": int(test_df["tss"].nunique()),
        "train_n": int(len(train_df)),
        "val_n": int(len(val_df)),
        "test_n": int(len(test_df)),
        "events_train": int(train_df["event_os"].sum()),
        "events_val": int(val_df["event_os"].sum()),
        "events_test": int(test_df["event_os"].sum()),
    }

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", out_dir)
    print(json.dumps(meta, indent=2))