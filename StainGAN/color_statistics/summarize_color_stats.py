#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd

RAW_CSV = "/work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/color_statistics/color_stats_raw.csv"
STAINGAN_CSV = "/work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/color_statistics/color_stats_staingan.csv"

OUT_DIR = Path("/work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/color_statistics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TSS_SUMMARY = OUT_DIR / "color_stats_by_tss.csv"
OUT_VARIANCE = OUT_DIR / "color_variance_by_tss.csv"
OUT_SLIDE_DISTANCE = OUT_DIR / "lab_distance_to_template.csv"
OUT_TEMPLATE = OUT_DIR / "bh_lab_template_summary.csv"

TARGET_TSS = "BH"

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


def compute_bh_template_vector(raw_df, target_tss=TARGET_TSS):
    bh_df = raw_df[raw_df["tss"].astype(str) == str(target_tss)].copy()
    if bh_df.empty:
        raise ValueError(f"No raw slides found for target TSS={target_tss}")

    vec = bh_df[LAB_COLS].mean().to_numpy(dtype=float)

    summary = pd.DataFrame([{
        "target_tss": target_tss,
        "n_slides_used": len(bh_df),
        "lab_mean_l": vec[0],
        "lab_mean_a": vec[1],
        "lab_mean_b": vec[2],
        "lab_std_l": vec[3],
        "lab_std_a": vec[4],
        "lab_std_b": vec[5],
        "template_source": f"raw_{target_tss}_aggregate_mean",
    }])
    summary.to_csv(OUT_TEMPLATE, index=False)

    return vec, f"raw_{target_tss}_aggregate_mean"


def main():
    raw_df = load_df(RAW_CSV, "raw")
    staingan_df = load_df(STAINGAN_CSV, "staingan")
    all_df = pd.concat([raw_df, staingan_df], ignore_index=True)

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

    if {"raw", "staingan"}.issubset(set(variance_df["stain_condition"])):
        raw_row = variance_df[variance_df["stain_condition"] == "raw"].iloc[0]
        staingan_row = variance_df[variance_df["stain_condition"] == "staingan"].iloc[0]

        delta = {"stain_condition": "delta_staingan_minus_raw", "n_tss": np.nan}
        pct_reduction = {"stain_condition": "percent_reduction_staingan_vs_raw", "n_tss": np.nan}

        for col in STAT_COLS:
            raw_var = raw_row[f"{col}_var_across_tss"]
            staingan_var = staingan_row[f"{col}_var_across_tss"]
            delta[f"{col}_var_across_tss"] = staingan_var - raw_var
            pct_reduction[f"{col}_var_across_tss"] = (
                100.0 * (raw_var - staingan_var) / raw_var if pd.notna(raw_var) and raw_var != 0 else np.nan
            )

        variance_df = pd.concat(
            [variance_df, pd.DataFrame([delta]), pd.DataFrame([pct_reduction])],
            ignore_index=True
        )

    variance_df.to_csv(OUT_VARIANCE, index=False)

    template_vec, template_source = compute_bh_template_vector(raw_df, target_tss=TARGET_TSS)

    dist_rows = []
    for df_name, df in [("raw", raw_df), ("staingan", staingan_df)]:
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
    print(f"Saved: {OUT_TEMPLATE}")
    print(f"Saved: {OUT_DIR / 'lab_distance_summary_by_tss.csv'}")
    print(f"template_source={template_source}")


if __name__ == "__main__":
    main()