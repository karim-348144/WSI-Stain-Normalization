#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image

RAW_CSV = "/homes/smdaniyal/stain_norm/color_statistics/color_stats_raw.csv"
MAC_CSV = "/homes/smdaniyal/stain_norm/color_statistics/color_stats_macenko.csv"

MACENKO_TARGET_IMG = "/work/ai4bio2025/stain_norm/targets/target_macenko.jpg"

OUT_DIR = Path("/homes/smdaniyal/stain_norm/color_statistics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TSS_SUMMARY = OUT_DIR / "color_stats_by_tss.csv"
OUT_VARIANCE = OUT_DIR / "color_variance_by_tss.csv"
OUT_SLIDE_DISTANCE = OUT_DIR / "lab_distance_to_template.csv"

STAT_COLS = [
    "rgb_mean_r", "rgb_mean_g", "rgb_mean_b",
    "rgb_std_r",  "rgb_std_g",  "rgb_std_b",
    "lab_mean_l", "lab_mean_a", "lab_mean_b",
    "lab_std_l",  "lab_std_a",  "lab_std_b",
]

LAB_COLS = [
    "lab_mean_l", "lab_mean_a", "lab_mean_b",
    "lab_std_l",  "lab_std_a",  "lab_std_b",
]

def load_df(path, stain_label):
    df = pd.read_csv(path)
    keep = ["slide_id", "tss", "stain_condition"] + [c for c in STAT_COLS if c in df.columns]
    df = df[[c for c in keep if c in df.columns]].copy()
    if "stain_condition" not in df.columns:
        df["stain_condition"] = stain_label
    return df

def load_template_vector():
    p = Path(MACENKO_TARGET_IMG)
    if not p.is_file():
        return None, "raw_global_mean"

    img = np.array(Image.open(p).convert("RGB"))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    flat_lab = lab.reshape(-1, 3)
    mu = flat_lab.mean(axis=0)
    sd = flat_lab.std(axis=0)

    vec = np.array([mu[0], mu[1], mu[2], sd[0], sd[1], sd[2]], dtype=float)
    return vec, "macenko_target_image"

def main():
    raw_df = load_df(RAW_CSV, "raw")
    mac_df = load_df(MAC_CSV, "macenko")
    all_df = pd.concat([raw_df, mac_df], ignore_index=True)

    tss_summary = (
        all_df
        .groupby(["stain_condition", "tss"])[STAT_COLS]
        .agg(["mean", "std"])
    )
    tss_summary.columns = [f"{col}_{agg}" for col, agg in tss_summary.columns]
    tss_summary = tss_summary.reset_index()
    tss_summary.to_csv(OUT_TSS_SUMMARY, index=False)

    tss_means = (
        all_df
        .groupby(["stain_condition", "tss"])[STAT_COLS]
        .mean()
        .reset_index()
    )

    variance_rows = []
    for stain_condition, subdf in tss_means.groupby("stain_condition"):
        row = {"stain_condition": stain_condition, "n_tss": subdf["tss"].nunique()}
        for col in STAT_COLS:
            row[f"{col}_var_across_tss"] = subdf[col].var(ddof=1)
        variance_rows.append(row)

    variance_df = pd.DataFrame(variance_rows)

    if {"raw", "macenko"}.issubset(set(variance_df["stain_condition"])):
        raw_row = variance_df[variance_df["stain_condition"] == "raw"].iloc[0]
        mac_row = variance_df[variance_df["stain_condition"] == "macenko"].iloc[0]

        delta = {"stain_condition": "delta_macenko_minus_raw", "n_tss": np.nan}
        pct_reduction = {"stain_condition": "percent_reduction_macenko_vs_raw", "n_tss": np.nan}

        for col in STAT_COLS:
            raw_var = raw_row[f"{col}_var_across_tss"]
            mac_var = mac_row[f"{col}_var_across_tss"]
            delta[f"{col}_var_across_tss"] = mac_var - raw_var
            pct_reduction[f"{col}_var_across_tss"] = (
                100.0 * (raw_var - mac_var) / raw_var if pd.notna(raw_var) and raw_var != 0 else np.nan
            )

        variance_df = pd.concat(
            [variance_df, pd.DataFrame([delta]), pd.DataFrame([pct_reduction])],
            ignore_index=True
        )

    variance_df.to_csv(OUT_VARIANCE, index=False)

    template_vec, template_source = load_template_vector()
    if template_vec is None:
        template_vec = raw_df[LAB_COLS].mean().to_numpy(dtype=float)

    dist_rows = []
    for df_name, df in [("raw", raw_df), ("macenko", mac_df)]:
        X = df[LAB_COLS].to_numpy(dtype=float)
        dists = np.linalg.norm(X - template_vec[None, :], axis=1)
        tmp = df[["slide_id", "tss"]].copy()
        tmp["stain_condition"] = df_name
        tmp["lab_distance_to_template"] = dists
        tmp["template_source"] = template_source
        dist_rows.append(tmp)

    dist_df = pd.concat(dist_rows, ignore_index=True)
    dist_df.to_csv(OUT_SLIDE_DISTANCE, index=False)

    dist_summary = (
        dist_df
        .groupby(["stain_condition", "tss"])["lab_distance_to_template"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    dist_summary.to_csv(OUT_DIR / "lab_distance_summary_by_tss.csv", index=False)

    print(f"Saved: {OUT_TSS_SUMMARY}")
    print(f"Saved: {OUT_VARIANCE}")
    print(f"Saved: {OUT_SLIDE_DISTANCE}")
    print(f"Saved: {OUT_DIR / 'lab_distance_summary_by_tss.csv'}")
    print(f"template_source={template_source}")

if __name__ == "__main__":
    main()