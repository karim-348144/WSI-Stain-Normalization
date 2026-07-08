#!/usr/bin/env python3
import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pycox.models.loss import CoxPHLoss

from models_abmil_transmil import ABMIL, TransMIL


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_feature_tensor(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    feats = obj["features"] if isinstance(obj, dict) else obj
    if not torch.is_tensor(feats):
        raise ValueError(f"{path} features are not a tensor")
    if feats.ndim != 2:
        raise ValueError(f"{path} expected 2D features, got shape {tuple(feats.shape)}")
    feats = feats.float()
    if not torch.isfinite(feats).all():
        raise ValueError(f"{path} contains non-finite feature values")
    return feats


def subsample_instances(x: torch.Tensor, max_patches: int, seed: int) -> torch.Tensor:
    if max_patches <= 0 or x.shape[0] <= max_patches:
        return x
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:max_patches]
    idx, _ = torch.sort(idx)
    return x[idx]


def cindex_from_risk(times, events, risks):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    risks = np.asarray(risks, dtype=float)

    mask = np.isfinite(times) & np.isfinite(events) & np.isfinite(risks)
    times = times[mask]
    events = events[mask]
    risks = risks[mask]

    num = 0.0
    den = 0.0
    n = len(times)
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = times[i], times[j]
            ei, ej = events[i], events[j]
            if ti == tj:
                continue
            if ei == 1 and ti < tj:
                den += 1.0
                if risks[i] > risks[j]:
                    num += 1.0
                elif risks[i] == risks[j]:
                    num += 0.5
            elif ej == 1 and tj < ti:
                den += 1.0
                if risks[j] > risks[i]:
                    num += 1.0
                elif risks[i] == risks[j]:
                    num += 0.5
    return float(num / den) if den > 0 else float("nan")


class BrcaBagDataset:
    def __init__(self, split_csv: str, feat_root: str, max_patches: int = 0, seed: int = 42):
        df = pd.read_csv(split_csv)

        expected_cols = {"slide_id", "time_os_months", "event_os"}
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {split_csv}: {missing}")

        self.slide_ids = df["slide_id"].astype(str).tolist()
        self.times = df["time_os_months"].astype("float32").to_numpy()
        self.events = df["event_os"].astype("float32").to_numpy()
        self.feat_root = feat_root
        self.max_patches = max_patches
        self.seed = seed

        missing_files = []
        for sid in self.slide_ids:
            p = os.path.join(self.feat_root, sid + ".pt")
            if not os.path.isfile(p):
                missing_files.append(p)
        if missing_files:
            preview = "\n".join(missing_files[:10])
            raise FileNotFoundError(f"Missing feature files under {feat_root}, examples:\n{preview}")

    def __len__(self):
        return len(self.slide_ids)

    def get_item(self, idx: int):
        sid = self.slide_ids[idx]
        path = os.path.join(self.feat_root, sid + ".pt")
        x = load_feature_tensor(path)
        x = subsample_instances(x, self.max_patches, seed=self.seed + idx)
        t = torch.tensor(self.times[idx], dtype=torch.float32)
        e = torch.tensor(self.events[idx], dtype=torch.float32)
        return x, t, e, sid


def get_in_dim(example_feat_path: str) -> int:
    arr = load_feature_tensor(example_feat_path)
    return int(arr.shape[1])


def make_model(model_name: str, in_dim: int, hidden_dim: int = 256, dropout: float = 0.25) -> nn.Module:
    if model_name.lower() == "abmil":
        backbone = ABMIL(in_dim=in_dim, hidden_dim=hidden_dim, attn_dim=128, dropout=dropout)
        out_dim = hidden_dim
    elif model_name.lower() == "transmil":
        backbone = TransMIL(in_dim=in_dim, d_model=hidden_dim, nhead=8, num_layers=2, dropout=dropout)
        out_dim = hidden_dim
    else:
        raise ValueError(f"Unknown model_name={model_name}")

    head = nn.Linear(out_dim, 1, bias=False)

    class SurvivalNet(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward_one_bag(self, x: torch.Tensor):
            z = self.backbone(x)
            if not torch.isfinite(z).all():
                raise RuntimeError("Non-finite bag embedding detected")
            risk = self.head(z).squeeze(-1)
            if not torch.isfinite(risk).all():
                raise RuntimeError("Non-finite risk detected")
            return risk

    return SurvivalNet(backbone, head)


def load_full_split(ds: BrcaBagDataset) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, List[str]]:
    xs, ts, es, sids = [], [], [], []
    for i in range(len(ds)):
        x, t, e, sid = ds.get_item(i)
        xs.append(x)
        ts.append(t)
        es.append(e)
        sids.append(sid)
    return xs, torch.stack(ts), torch.stack(es), sids


def sort_by_time_desc(xs, t, e, sids):
    order = torch.argsort(t, descending=True)
    xs = [xs[i] for i in order.tolist()]
    t = t[order]
    e = e[order]
    sids = [sids[i] for i in order.tolist()]
    return xs, t, e, sids


def compute_risks_streaming(net, xs: List[torch.Tensor], device: str) -> torch.Tensor:
    risks = []
    for x in xs:
        x = x.to(device, non_blocking=True)
        r = net.forward_one_bag(x)
        risks.append(r)
        del x
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    risk = torch.cat(risks, dim=0)
    if not torch.isfinite(risk).all():
        raise RuntimeError("Non-finite concatenated risks detected")
    return risk


def run_full_epoch(net, xs_train, t_train, e_train, optimizer, loss_fn, device, grad_clip: float):
    net.train()
    optimizer.zero_grad(set_to_none=True)

    t_train = t_train.to(device, non_blocking=True)
    e_train = e_train.to(device, non_blocking=True)

    risk = compute_risks_streaming(net, xs_train, device)
    loss = loss_fn(risk, t_train, e_train)

    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite full-batch Cox loss detected")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)

    for name, p in net.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise RuntimeError(f"Non-finite gradient in parameter {name}")

    optimizer.step()

    for name, p in net.named_parameters():
        if not torch.isfinite(p).all():
            raise RuntimeError(f"Non-finite parameter after optimizer step: {name}")

    return float(loss.item())


@torch.no_grad()
def predict_full_split(net, xs, t, e, sids, device):
    net.eval()
    risk = compute_risks_streaming(net, xs, device).detach().cpu().numpy()
    return risk, t.numpy(), e.numpy(), sids


def train_one(
    model_name: str,
    feat_root: str,
    split_dir: str,
    out_dir: str,
    lr: float = 1e-5,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    patience: int = 10,
    hidden_dim: int = 256,
    dropout: float = 0.25,
    max_patches: int = 0,
    seed: int = 42,
    device: str = "cuda",
    grad_clip: float = 1.0,
) -> Dict:
    seed_everything(seed)
    os.makedirs(out_dir, exist_ok=True)

    train_csv = os.path.join(split_dir, "train.csv")
    val_csv = os.path.join(split_dir, "val.csv")
    test_csv = os.path.join(split_dir, "test.csv")

    ds_train = BrcaBagDataset(train_csv, feat_root, max_patches=max_patches, seed=seed)
    ds_val = BrcaBagDataset(val_csv, feat_root, max_patches=max_patches, seed=seed)
    ds_test = BrcaBagDataset(test_csv, feat_root, max_patches=max_patches, seed=seed)

    example_sid = ds_train.slide_ids[0]
    in_dim = get_in_dim(os.path.join(feat_root, example_sid + ".pt"))

    net = make_model(model_name=model_name, in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = CoxPHLoss()

    xs_train, t_train, e_train, sid_train = load_full_split(ds_train)
    xs_val, t_val, e_val, sid_val = load_full_split(ds_val)
    xs_test, t_test, e_test, sid_test = load_full_split(ds_test)

    xs_train, t_train, e_train, sid_train = sort_by_time_desc(xs_train, t_train, e_train, sid_train)
    xs_val, t_val, e_val, sid_val = sort_by_time_desc(xs_val, t_val, e_val, sid_val)
    xs_test, t_test, e_test, sid_test = sort_by_time_desc(xs_test, t_test, e_test, sid_test)

    print(
        f"model={model_name} in_dim={in_dim} hidden_dim={hidden_dim} "
        f"max_patches={max_patches} full_batch_train={len(ds_train)} "
        f"val={len(ds_val)} test={len(ds_test)} lr={lr}"
    )

    best_val_cindex = -1.0
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = run_full_epoch(
            net, xs_train, t_train, e_train, optimizer, loss_fn, device, grad_clip=grad_clip
        )

        val_risk, val_t, val_e, _ = predict_full_split(net, xs_val, t_val, e_val, sid_val, device)
        val_cindex = cindex_from_risk(val_t, val_e, val_risk)

        row = {"epoch": epoch, "train_loss": float(train_loss), "val_cindex": float(val_cindex)}
        history.append(row)
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_cindex={val_cindex:.4f}")

        if np.isfinite(val_cindex) and val_cindex > best_val_cindex:
            best_val_cindex = float(val_cindex)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            torch.save(best_state, os.path.join(out_dir, "best_model.pt"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"early_stopping at epoch={epoch} best_epoch={best_epoch} best_val_cindex={best_val_cindex:.4f}")
            break

    if best_state is None:
        raise RuntimeError("Training ended without a valid finite validation model.")

    net.load_state_dict(best_state)

    test_risk, test_t, test_e, test_sid = predict_full_split(net, xs_test, t_test, e_test, sid_test, device)
    test_cindex = cindex_from_risk(test_t, test_e, test_risk)

    pd.DataFrame({
        "slide_id": test_sid,
        "time_os_months": test_t,
        "event_os": test_e,
        "risk": test_risk,
    }).to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)

    metrics = {
        "model_name": model_name,
        "feat_root": feat_root,
        "split_dir": split_dir,
        "seed": seed,
        "in_dim": in_dim,
        "train_n": len(ds_train),
        "val_n": len(ds_val),
        "test_n": len(ds_test),
        "hidden_dim": hidden_dim,
        "max_patches": max_patches,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "best_epoch": best_epoch,
        "best_val_cindex": float(best_val_cindex),
        "test_cindex": float(test_cindex),
        "test_risk_std": float(np.std(test_risk)),
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("FINAL_METRICS")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", choices=["abmil", "transmil"], required=True)
    ap.add_argument("--feat_root", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--max_patches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    args = ap.parse_args()

    train_one(
        model_name=args.model_name,
        feat_root=args.feat_root,
        split_dir=args.split_dir,
        out_dir=args.out_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        max_patches=args.max_patches,
        seed=args.seed,
        device=args.device,
        grad_clip=args.grad_clip,
    )


if __name__ == "__main__":
    main()
