#!/usr/bin/env python3
"""Train a weighted MLP for SignLanguage upper-body rotation plus translation correction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train_signlanguage_global_correction import build_model, import_torch, scalar_from_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--unweighted-loss", action="store_true")
    return parser.parse_args()


def split_arrays(data, split_id: int) -> tuple[np.ndarray, np.ndarray]:
    mask = data["split"] == split_id
    return data["x"][mask], data["y"][mask]


def make_loader(torch, DataLoader, TensorDataset, x, y, batch_size: int, shuffle: bool):
    return DataLoader(
        TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def weighted_smooth_l1(torch, pred, target, weight):
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weight.reshape(1, -1)).mean(dim=1)


def eval_split(torch, model, loader, device, weight) -> dict[str, float]:
    model.eval()
    losses = []
    unweighted_losses = []
    rot_l2 = []
    transl_l2 = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            losses.append(weighted_smooth_l1(torch, pred, y, weight).detach().cpu().numpy())
            unweighted_losses.append(torch.nn.functional.smooth_l1_loss(pred, y, reduction="none").mean(dim=1).detach().cpu().numpy())
            if pred.shape[1] > 3:
                rot_l2.append(torch.linalg.norm((pred[:, :-3] - y[:, :-3]).reshape(-1, 3), dim=1).detach().cpu().numpy())
            transl_l2.append(torch.linalg.norm(pred[:, -3:] - y[:, -3:], dim=1).detach().cpu().numpy())
    if not losses:
        return {
            "loss": float("nan"),
            "unweighted_loss": float("nan"),
            "rot_l2_deg": float("nan"),
            "transl_l2_m": float("nan"),
        }
    return {
        "loss": float(np.concatenate(losses).mean()),
        "unweighted_loss": float(np.concatenate(unweighted_losses).mean()),
        "rot_l2_deg": float(np.degrees(np.concatenate(rot_l2)).mean()) if rot_l2 else float("nan"),
        "transl_l2_m": float(np.concatenate(transl_l2).mean()),
    }


def main() -> None:
    args = parse_args()
    torch, nn, DataLoader, TensorDataset = import_torch()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    data = np.load(args.dataset, allow_pickle=True)
    x_train, y_train = split_arrays(data, 0)
    x_val, y_val = split_arrays(data, 1)
    x_test, y_test = split_arrays(data, 2)
    if len(x_train) == 0:
        raise ValueError("Dataset has no train samples")
    if len(x_val) == 0:
        x_val, y_val = x_train, y_train

    if "target_weight" in data.files and not args.unweighted_loss:
        weight_np = np.asarray(data["target_weight"], dtype=np.float32).reshape(-1)
    else:
        weight_np = np.ones(y_train.shape[1], dtype=np.float32)
    if weight_np.shape[0] != y_train.shape[1]:
        raise ValueError(f"target_weight has {weight_np.shape[0]} values, expected {y_train.shape[1]}")
    weight = torch.from_numpy(weight_np).to(device)

    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std if len(x_test) else x_test

    train_loader = make_loader(torch, DataLoader, TensorDataset, x_train, y_train, args.batch_size, True)
    val_loader = make_loader(torch, DataLoader, TensorDataset, x_val, y_val, args.batch_size, False)
    test_loader = make_loader(torch, DataLoader, TensorDataset, x_test, y_test, args.batch_size, False) if len(x_test) else None

    model = build_model(nn, x_train.shape[1], args.hidden_dim, args.layers, args.dropout).to(device)
    if hasattr(model[-1], "in_features"):
        model[-1] = nn.Linear(model[-1].in_features, y_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    history = []
    best_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_path = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        step_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = weighted_smooth_l1(torch, pred, y, weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        train_metrics = eval_split(torch, model, train_loader, device, weight)
        val_metrics = eval_split(torch, model, val_loader, device, weight)
        row = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_step_loss": float(np.mean(step_losses)),
            "train_loss": train_metrics["loss"],
            "train_unweighted_loss": train_metrics["unweighted_loss"],
            "train_rot_l2_deg": train_metrics["rot_l2_deg"],
            "train_transl_l2_m": train_metrics["transl_l2_m"],
            "val_loss": val_metrics["loss"],
            "val_unweighted_loss": val_metrics["unweighted_loss"],
            "val_rot_l2_deg": val_metrics["rot_l2_deg"],
            "val_transl_l2_m": val_metrics["transl_l2_m"],
        }
        history.append(row)
        print(
            f"epoch {epoch:04d} train={row['train_loss']:.6f} val={row['val_loss']:.6f} "
            f"rot={row['val_rot_l2_deg']:.2f}deg transl={row['val_transl_l2_m']:.4f}m"
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
                    "output_dim": int(y_train.shape[1]),
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "dropout": args.dropout,
                    "feature_preset": str(scalar_from_npz(data, "feature_preset", "upper_global")),
                    "temporal_radius": int(scalar_from_npz(data, "temporal_radius", 1)),
                    "target_mode": str(scalar_from_npz(data, "target_mode", "torso_arms")),
                    "target_kind": str(scalar_from_npz(data, "target_kind", "upperbody_transl")),
                    "target_body_indices": scalar_from_npz(data, "target_body_indices", []),
                    "target_smplx_joints": scalar_from_npz(data, "target_smplx_joints", []),
                    "target_weight": weight_np.astype(float).tolist(),
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
        "train": eval_split(torch, model, train_loader, device, weight),
        "val": eval_split(torch, model, val_loader, device, weight),
    }
    if test_loader is not None:
        final_metrics["test"] = eval_split(torch, model, test_loader, device, weight)
    report = {
        "best_model": str(best_path.resolve()),
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "samples": {"train": int(len(x_train)), "val": int(len(x_val)), "test": int(len(x_test))},
        "metrics": final_metrics,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
