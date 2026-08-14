#!/usr/bin/env python3
"""Train a small MLP for SignLanguage global-orient/translation correction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--orient-loss-weight", type=float, default=1.0)
    parser.add_argument("--transl-loss-weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def import_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit("PyTorch is required for training. Run inside video2smplx_shared310.") from exc
    return torch, nn, DataLoader, TensorDataset


def build_model(nn, input_dim: int, hidden_dim: int, layers: int, dropout: float):
    modules = []
    dim = input_dim
    for _ in range(max(1, layers)):
        modules.extend(
            [
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        dim = hidden_dim
    modules.append(nn.Linear(dim, 6))
    return nn.Sequential(*modules)


def split_arrays(data, split_id: int) -> tuple[np.ndarray, np.ndarray]:
    mask = data["split"] == split_id
    return data["x"][mask], data["y"][mask]


def scalar_from_npz(data, key: str, default):
    if key not in data.files:
        return default
    value = data[key]
    if value.shape == ():
        return value.item()
    return value.tolist()


def make_loader(torch, DataLoader, TensorDataset, x, y, batch_size: int, shuffle: bool, workers: int):
    dataset = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False)


def eval_split(torch, model, loader, device, orient_weight: float, transl_weight: float) -> dict[str, float]:
    model.eval()
    losses = []
    orient_errors = []
    transl_errors = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            orient_loss = torch.nn.functional.smooth_l1_loss(pred[:, :3], y[:, :3], reduction="none").mean(dim=1)
            transl_loss = torch.nn.functional.smooth_l1_loss(pred[:, 3:], y[:, 3:], reduction="none").mean(dim=1)
            loss = orient_weight * orient_loss + transl_weight * transl_loss
            losses.append(loss.detach().cpu().numpy())
            orient_errors.append(torch.linalg.norm(pred[:, :3] - y[:, :3], dim=1).detach().cpu().numpy())
            transl_errors.append(torch.linalg.norm(pred[:, 3:] - y[:, 3:], dim=1).detach().cpu().numpy())
    if not losses:
        return {"loss": float("nan"), "orient_l2_deg": float("nan"), "transl_l2_m": float("nan")}
    return {
        "loss": float(np.concatenate(losses).mean()),
        "orient_l2_deg": float(np.degrees(np.concatenate(orient_errors)).mean()),
        "transl_l2_m": float(np.concatenate(transl_errors).mean()),
    }


def main() -> None:
    args = parse_args()
    torch, nn, DataLoader, TensorDataset = import_torch()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    data = np.load(args.dataset, allow_pickle=True)
    x_train, y_train = split_arrays(data, 0)
    x_val, y_val = split_arrays(data, 1)
    x_test, y_test = split_arrays(data, 2)
    if len(x_train) == 0:
        raise ValueError("Dataset has no train samples")
    if len(x_val) == 0:
        x_val, y_val = x_train, y_train

    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std if len(x_test) else x_test

    train_loader = make_loader(torch, DataLoader, TensorDataset, x_train, y_train, args.batch_size, True, args.num_workers)
    val_loader = make_loader(torch, DataLoader, TensorDataset, x_val, y_val, args.batch_size, False, args.num_workers)
    test_loader = make_loader(torch, DataLoader, TensorDataset, x_test, y_test, args.batch_size, False, args.num_workers) if len(x_test) else None

    model = build_model(nn, x_train.shape[1], args.hidden_dim, args.layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    best_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            orient_loss = torch.nn.functional.smooth_l1_loss(pred[:, :3], y[:, :3])
            transl_loss = torch.nn.functional.smooth_l1_loss(pred[:, 3:], y[:, 3:])
            loss = args.orient_loss_weight * orient_loss + args.transl_loss_weight * transl_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        train_metrics = eval_split(torch, model, train_loader, device, args.orient_loss_weight, args.transl_loss_weight)
        val_metrics = eval_split(torch, model, val_loader, device, args.orient_loss_weight, args.transl_loss_weight)
        row = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss_step_mean": float(np.mean(train_losses)),
            "train_loss": train_metrics["loss"],
            "train_orient_l2_deg": train_metrics["orient_l2_deg"],
            "train_transl_l2_m": train_metrics["transl_l2_m"],
            "val_loss": val_metrics["loss"],
            "val_orient_l2_deg": val_metrics["orient_l2_deg"],
            "val_transl_l2_m": val_metrics["transl_l2_m"],
        }
        history.append(row)
        print(
            f"epoch {epoch:04d} train={row['train_loss']:.6f} "
            f"val={row['val_loss']:.6f} orient={row['val_orient_l2_deg']:.2f}deg "
            f"transl={row['val_transl_l2_m']:.4f}m"
        )

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_mean": mean.astype(float).tolist(),
                    "feature_std": std.astype(float).tolist(),
                    "input_dim": int(x_train.shape[1]),
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "dropout": args.dropout,
                    "feature_preset": str(scalar_from_npz(data, "feature_preset", "upper_global")),
                    "temporal_radius": int(scalar_from_npz(data, "temporal_radius", 1)),
                    "dataset": str(args.dataset.resolve()),
                    "epoch": epoch,
                    "val_loss": best_loss,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                print(f"early stop at epoch {epoch}; best epoch {best_epoch}")
                break

    with (args.output_dir / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    final_metrics = {
        "train": eval_split(torch, model, train_loader, device, args.orient_loss_weight, args.transl_loss_weight),
        "val": eval_split(torch, model, val_loader, device, args.orient_loss_weight, args.transl_loss_weight),
    }
    if test_loader is not None:
        final_metrics["test"] = eval_split(torch, model, test_loader, device, args.orient_loss_weight, args.transl_loss_weight)
    report = {
        "dataset": str(args.dataset.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "best_model": str(best_path.resolve()),
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "samples": {
            "train": int(len(x_train)),
            "val": int(len(x_val)),
            "test": int(len(x_test)),
        },
        "metrics": final_metrics,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
