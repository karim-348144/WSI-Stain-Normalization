import pandas as pd
from pathlib import Path
import json

MANIFEST = "/homes/smdaniyal/stain_norm/manifests/brca_matching_slide_tss_manifest.csv"
OUT_DIR = Path("/homes/smdaniyal/stain_norm/splits/brca_tss_holdout_E2_BH")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_TSS = "E2"
VAL_TSS = "BH"

df = pd.read_csv(MANIFEST)

test_df = df[df["tss"] == TEST_TSS].copy()
val_df = df[df["tss"] == VAL_TSS].copy()
train_df = df[(df["tss"] != TEST_TSS) & (df["tss"] != VAL_TSS)].copy()

keep_cols = ["slide_id", "patient_id", "time_os_months", "event_os", "tss"]

train_df[keep_cols].to_csv(OUT_DIR / "train.csv", index=False)
val_df[keep_cols].to_csv(OUT_DIR / "val.csv", index=False)
test_df[keep_cols].to_csv(OUT_DIR / "test.csv", index=False)

meta = {
    "split_type": "unseen_center_holdout",
    "test_tss": TEST_TSS,
    "val_tss": VAL_TSS,
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

with open(OUT_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved splits to:", OUT_DIR)
print(json.dumps(meta, indent=2))
