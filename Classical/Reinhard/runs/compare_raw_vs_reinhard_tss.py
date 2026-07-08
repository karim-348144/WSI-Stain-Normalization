#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

RUNS_ROOT = Path("/homes/wdkarim/runs")
COMPARISON_ROOT = Path("/homes/wdkarim/Comparisons")
OUT_METRICS = COMPARISON_ROOT / "raw_vs_reinhard_tss_metrics.csv"
OUT_COMPARISON = COMPARISON_ROOT / "raw_vs_reinhard_tss_comparison.csv"

HOLDOUTS = [
    "tss_holdout_A2_BH",
    "tss_holdout_A2_E2",
    "tss_holdout_BH_A2",
    "tss_holdout_BH_E2",
    "tss_holdout_E2_A2",
    "tss_holdout_E2_BH",
]

RUN_MAP = []
for h in HOLDOUTS:
    RUN_MAP.extend([
        {"holdout": h, "model": "abmil",    "feature_set": "raw",      "run_dir": RUNS_ROOT / h / "abmil_raw"},
        {"holdout": h, "model": "abmil",    "feature_set": "reinhard", "run_dir": RUNS_ROOT / h / "abmil_reinhard"},
        {"holdout": h, "model": "transmil", "feature_set": "raw",      "run_dir": RUNS_ROOT / h / "transmil_raw"},
        {"holdout": h, "model": "transmil", "feature_set": "reinhard", "run_dir": RUNS_ROOT / h / "transmil_reinhard"},
    ])

rows = []
missing = []

for item in RUN_MAP:
    mpath = item["run_dir"] / "metrics.json"
    if not mpath.is_file():
        missing.append(str(mpath))
        continue
    with open(mpath) as f:
        m = json.load(f)
    rows.append({
        "holdout": item["holdout"],
        "model": item["model"],
        "feature_set": item["feature_set"],
        "best_epoch": m.get("best_epoch"),
        "best_val_cindex": m.get("best_val_cindex"),
        "test_cindex": m.get("test_cindex"),
        "train_n": m.get("train_n"),
        "val_n": m.get("val_n"),
        "test_n": m.get("test_n"),
        "feat_root": m.get("feat_root"),
        "split_dir": m.get("split_dir"),
        "run_dir": str(item["run_dir"]),
    })

df = pd.DataFrame(rows)
if df.empty:
    print("No metrics found")
    if missing:
        print("\nMissing metrics paths:")
        for p in missing:
            print(p)
    raise SystemExit(1)

df = df.sort_values(["holdout", "model", "feature_set"]).reset_index(drop=True)
df.to_csv(OUT_METRICS, index=False)

pivot = df.pivot_table(
    index=["holdout", "model"],
    columns="feature_set",
    values=["best_val_cindex", "test_cindex"],
    aggfunc="first"
)

pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
pivot = pivot.reset_index()

if "test_cindex_reinhard" in pivot.columns and "test_cindex_raw" in pivot.columns:
    pivot["delta_test_cindex_reinhard_minus_raw"] = pivot["test_cindex_reinhard"] - pivot["test_cindex_raw"]

if "best_val_cindex_reinhard" in pivot.columns and "best_val_cindex_raw" in pivot.columns:
    pivot["delta_val_cindex_reinhard_minus_raw"] = pivot["best_val_cindex_reinhard"] - pivot["best_val_cindex_raw"]

pivot.to_csv(OUT_COMPARISON, index=False)

print("\nPer-run metrics:")
print(df.to_string(index=False))

print("\nRaw vs reinhard comparison:")
print(pivot.to_string(index=False))

if missing:
    print("\nMissing metrics paths:")
    for p in missing:
        print(p)