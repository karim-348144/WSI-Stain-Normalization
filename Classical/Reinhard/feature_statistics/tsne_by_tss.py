#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

INPUT_CSV = "/homes/wdkarim/feature_statistics/slide_embeddings_raw_reinhard.csv"
OUT_DIR = Path("/homes/wdkarim/feature_statistics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TSNE_CSV = OUT_DIR / "tsne_by_tss.csv"
OUT_RAW_PNG = OUT_DIR / "tsne_raw_tss.png"
OUT_REI_PNG = OUT_DIR / "tsne_reinhard_tss.png"
OUT_COMBINED_PNG = OUT_DIR / "tsne_raw_vs_reinhard_tss.png"

RANDOM_STATE = 42
PERPLEXITY = 30
TOP_TSS = 12


def prepare_labels(df: pd.DataFrame):
    counts = df["tss"].value_counts()
    top_tss = set(counts.head(TOP_TSS).index)
    df = df.copy()
    df["tss_plot"] = df["tss"].where(df["tss"].isin(top_tss), "Other")
    return df


def compute_tsne(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c.startswith("f_")]
    X = df[feat_cols].to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        init="pca",
        learning_rate="auto",
        random_state=RANDOM_STATE,
        metric="euclidean",
    )
    Z = tsne.fit_transform(X)

    out = df.copy()
    out["tsne_1"] = Z[:, 0]
    out["tsne_2"] = Z[:, 1]
    return out


def make_scatter(ax, df, title):
    labels = sorted(df["tss_plot"].unique())
    cmap = plt.get_cmap("tab20")

    for i, label in enumerate(labels):
        sub = df[df["tss_plot"] == label]
        ax.scatter(
            sub["tsne_1"],
            sub["tsne_2"],
            s=14,
            alpha=0.75,
            label=label,
            color=cmap(i % 20),
            edgecolors="none",
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")


def main():
    df = pd.read_csv(INPUT_CSV)

    required = {"slide_id", "tss", "stain_condition"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = prepare_labels(df)
    tsne_df = compute_tsne(df)
    tsne_df.to_csv(OUT_TSNE_CSV, index=False)

    raw_df = tsne_df[tsne_df["stain_condition"] == "raw"].copy()
    rei_df = tsne_df[tsne_df["stain_condition"] == "reinhard"].copy()

    # Raw plot
    fig, ax = plt.subplots(figsize=(10, 8))
    make_scatter(ax, raw_df, "UNI t-SNE by TSS (Raw)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_RAW_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Reinhard plot
    fig, ax = plt.subplots(figsize=(10, 8))
    make_scatter(ax, rei_df, "UNI t-SNE by TSS (Reinhard)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_REI_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Combined plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=False, sharey=False)
    make_scatter(axes[0], raw_df, "Raw")
    make_scatter(axes[1], rei_df, "Reinhard")

    handles, labels = axes[1].get_legend_handles_labels()
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    fig.suptitle("UNI t-SNE by TSS (Raw vs Reinhard)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    fig.savefig(OUT_COMBINED_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_TSNE_CSV}")
    print(f"Saved: {OUT_RAW_PNG}")
    print(f"Saved: {OUT_REI_PNG}")
    print(f"Saved: {OUT_COMBINED_PNG}")


if __name__ == "__main__":
    main()