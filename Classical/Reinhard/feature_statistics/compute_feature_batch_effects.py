
#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MANIFEST = "/homes/wdkarim/manifests/brca_matching_slide_tss_manifest.csv"

RAW_FEAT_ROOT = "/homes/wdkarim/features/raw_uni"
REI_FEAT_ROOT = "/homes/wdkarim/features/reinhard_uni_all"

OUT_DIR = Path("/homes/wdkarim/feature_statistics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SLIDE_EMB = OUT_DIR / "slide_embeddings_raw_reinhard.csv"
OUT_BATCH_METRICS = OUT_DIR / "feature_batch_effect_metrics.csv"
OUT_TSS_COUNTS = OUT_DIR / "feature_batch_effect_tss_counts.csv"

MIN_SAMPLES_PER_TSS = 2
KNN_NEIGHBORS = 5
CV_FOLDS = 5


def load_manifest():
    df = pd.read_csv(MANIFEST)
    required = {"slide_id", "tss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    return df[["slide_id", "tss"]].copy()


def load_feature_mean(pt_path: Path):
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    feats = obj["features"] if isinstance(obj, dict) else obj
    if not torch.is_tensor(feats):
        raise ValueError(f"{pt_path} features are not a tensor")
    if feats.ndim != 2:
        raise ValueError(f"{pt_path} expected 2D features, got {tuple(feats.shape)}")
    feats = feats.float()
    if not torch.isfinite(feats).all():
        raise ValueError(f"{pt_path} contains non-finite features")
    return feats.mean(dim=0).numpy(), int(feats.shape[0]), int(feats.shape[1])


def build_slide_embedding_df(feat_root: str, stain_condition: str, manifest_df: pd.DataFrame):
    feat_root = Path(feat_root)
    rows = []

    for _, row in manifest_df.iterrows():
        slide_id = str(row["slide_id"])
        tss = str(row["tss"])
        pt_path = feat_root / f"{slide_id}.pt"

        if not pt_path.is_file():
            continue

        mean_vec, num_patches, feat_dim = load_feature_mean(pt_path)

        out = {
            "slide_id": slide_id,
            "tss": tss,
            "stain_condition": stain_condition,
            "num_patches": num_patches,
            "feat_dim": feat_dim,
        }
        for i, v in enumerate(mean_vec):
            out[f"f_{i}"] = float(v)
        rows.append(out)

    return pd.DataFrame(rows)


def compute_metrics(embed_df: pd.DataFrame, stain_condition: str):
    feat_cols = [c for c in embed_df.columns if c.startswith("f_")]

    valid_tss = embed_df["tss"].value_counts()
    valid_tss = valid_tss[valid_tss >= MIN_SAMPLES_PER_TSS].index
    df = embed_df[embed_df["tss"].isin(valid_tss)].copy()

    X = df[feat_cols].to_numpy(dtype=float)
    y = df["tss"].astype(str).to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    sil = silhouette_score(Xs, y, metric="euclidean")

    min_class_count = pd.Series(y).value_counts().min()
    n_splits = min(CV_FOLDS, int(min_class_count))
    if n_splits < 2:
        raise ValueError(f"Not enough samples per TSS for CV in {stain_condition}")

    clf = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    metrics = {
        "stain_condition": stain_condition,
        "n_slides_total": int(len(embed_df)),
        "n_slides_used": int(len(df)),
        "n_tss_total": int(embed_df["tss"].nunique()),
        "n_tss_used": int(df["tss"].nunique()),
        "embedding_dim": int(len(feat_cols)),
        "silhouette_tss": float(sil),
        "knn_tss_acc_mean": float(acc_scores.mean()),
        "knn_tss_acc_std": float(acc_scores.std(ddof=1)) if len(acc_scores) > 1 else 0.0,
        "knn_neighbors": int(KNN_NEIGHBORS),
        "cv_folds": int(n_splits),
        "min_samples_per_tss": int(MIN_SAMPLES_PER_TSS),
    }
    return metrics, df


def main():
    manifest_df = load_manifest()

    raw_df = build_slide_embedding_df(RAW_FEAT_ROOT, "raw", manifest_df)
    rei_df = build_slide_embedding_df(REI_FEAT_ROOT, "reinhard", manifest_df)

    all_df = pd.concat([raw_df, rei_df], ignore_index=True)
    all_df.to_csv(OUT_SLIDE_EMB, index=False)

    raw_metrics, raw_used = compute_metrics(raw_df, "raw")
    rei_metrics, rei_used = compute_metrics(rei_df, "reinhard")

    metrics_df = pd.DataFrame([raw_metrics, rei_metrics])

    if {"raw", "reinhard"}.issubset(set(metrics_df["stain_condition"])):
        mraw = metrics_df[metrics_df["stain_condition"] == "raw"].iloc[0]
        mrei = metrics_df[metrics_df["stain_condition"] == "reinhard"].iloc[0]

        delta_row = {
            "stain_condition": "delta_reinhard_minus_raw",
            "n_slides_total": np.nan,
            "n_slides_used": np.nan,
            "n_tss_total": np.nan,
            "n_tss_used": np.nan,
            "embedding_dim": np.nan,
            "silhouette_tss": mrei["silhouette_tss"] - mraw["silhouette_tss"],
            "knn_tss_acc_mean": mrei["knn_tss_acc_mean"] - mraw["knn_tss_acc_mean"],
            "knn_tss_acc_std": mrei["knn_tss_acc_std"] - mraw["knn_tss_acc_std"],
            "knn_neighbors": np.nan,
            "cv_folds": np.nan,
            "min_samples_per_tss": np.nan,
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([delta_row])], ignore_index=True)

        pct_row = {
            "stain_condition": "percent_change_reinhard_vs_raw",
            "n_slides_total": np.nan,
            "n_slides_used": np.nan,
            "n_tss_total": np.nan,
            "n_tss_used": np.nan,
            "embedding_dim": np.nan,
            "silhouette_tss": (
                100.0 * (mrei["silhouette_tss"] - mraw["silhouette_tss"]) / mraw["silhouette_tss"]
                if mraw["silhouette_tss"] != 0 else np.nan
            ),
            "knn_tss_acc_mean": (
                100.0 * (mrei["knn_tss_acc_mean"] - mraw["knn_tss_acc_mean"]) / mraw["knn_tss_acc_mean"]
                if mraw["knn_tss_acc_mean"] != 0 else np.nan
            ),
            "knn_tss_acc_std": np.nan,
            "knn_neighbors": np.nan,
            "cv_folds": np.nan,
            "min_samples_per_tss": np.nan,
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([pct_row])], ignore_index=True)

    metrics_df.to_csv(OUT_BATCH_METRICS, index=False)

    tss_counts = pd.concat([
        raw_used.groupby("tss").size().rename("n_slides_raw"),
        rei_used.groupby("tss").size().rename("n_slides_reinhard")
    ], axis=1).reset_index()
    tss_counts.to_csv(OUT_TSS_COUNTS, index=False)

    print(f"Saved: {OUT_SLIDE_EMB}")
    print(f"Saved: {OUT_BATCH_METRICS}")
    print(f"Saved: {OUT_TSS_COUNTS}")

    print("\nFeature batch-effect metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()


