from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banana_ripeness.data.dataset import MultiTaskDataset
from banana_ripeness.inference.preprocess import build_transforms
from banana_ripeness.modeling.model import build_model
from banana_ripeness.modeling.train import run_epoch
from banana_ripeness.utils.io import ensure_dir, load_yaml
from banana_ripeness.utils.logger import setup_logger
from banana_ripeness.utils.seed import set_seed


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_metric: float, train_cfg: dict, data_cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "train_config": train_cfg,
        "data_config": data_cfg,
        "classes_ripeness": data_cfg.get("classes_ripeness", []),
        "classes_defect": data_cfg.get("classes_defect", []),
        "img_size": train_cfg.get("img_size", 224),
        "model_name": train_cfg.get("model_name", "efficientnet_b0"),
    }
    torch.save(payload, path)

def is_shared_memory_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return "Couldn't open shared file mapping" in message or "error code: 1455" in message


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-task model")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to train config")
    parser.add_argument("--metadata", default=None, help="Override metadata CSV path")
    args = parser.parse_args()

    train_cfg_path = Path(args.config)
    if not train_cfg_path.is_absolute():
        train_cfg_path = ROOT / train_cfg_path
    train_cfg = load_yaml(train_cfg_path)

    data_cfg_path = Path(train_cfg.get("data_config", "configs/data.yaml"))
    if not data_cfg_path.is_absolute():
        data_cfg_path = ROOT / data_cfg_path
    data_cfg = load_yaml(data_cfg_path)

    seed = int(data_cfg.get("seed", 42))
    set_seed(seed)

    logger = setup_logger("train")

    if args.metadata:
        metadata_path = Path(args.metadata)
        if not metadata_path.is_absolute():
            metadata_path = ROOT / metadata_path
    else:
        metadata_path = ROOT / "data" / "processed" / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}. Run 02_prepare_dataset.py first.")

    classes_ripeness = data_cfg.get("classes_ripeness", [])
    classes_defect = data_cfg.get("classes_defect", [])

    img_size = int(train_cfg.get("img_size", 224))
    batch_size = int(train_cfg.get("batch_size", 32))
    if os.name == "nt":
        if "num_workers" in train_cfg:
            requested_workers = int(train_cfg.get("num_workers", 0))
        else:
            requested_workers = 0
        if requested_workers <= 0:
            num_workers = 0
        else:
            num_workers = 1 if requested_workers > 1 else requested_workers
    else:
        num_workers = int(train_cfg.get("num_workers", 4))

    augment_cfg = train_cfg.get("augment", {}) or {}
    train_tf = build_transforms(img_size, is_train=True, augment_cfg=augment_cfg)
    val_tf = build_transforms(img_size, is_train=False)

    train_dataset = MultiTaskDataset(
        metadata_path,
        root_dir=ROOT,
        split="train",
        classes_ripeness=classes_ripeness,
        classes_defect=classes_defect,
        transform=train_tf,
    )
    val_dataset = MultiTaskDataset(
        metadata_path,
        root_dir=ROOT,
        split="val",
        classes_ripeness=classes_ripeness,
        classes_defect=classes_defect,
        transform=val_tf,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_loaders(worker_count: int, sampler=None) -> tuple[DataLoader, DataLoader, bool]:
        pin_memory = device.type == "cuda" and worker_count <= 1
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": worker_count,
            "pin_memory": pin_memory,
        }
        if worker_count > 0:
            loader_kwargs["prefetch_factor"] = 2
            loader_kwargs["persistent_workers"] = False if os.name == "nt" else True

        train_loader = DataLoader(
            train_dataset,
            shuffle=True if sampler is None else False,
            sampler=sampler,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs,
        )
        return train_loader, val_loader, pin_memory

    logger.info("Train samples: %d | Val samples: %d", len(train_dataset), len(val_dataset))

    use_class_weights = bool(train_cfg.get("use_class_weights", True))
    use_defect_pos_weight = bool(train_cfg.get("use_defect_pos_weight", True))

    ripeness_class_weights = None
    defect_pos_weight = None
    if use_class_weights:
        ripeness_df = train_dataset.df[
            train_dataset.df["label_ripeness"].notna() & (train_dataset.df["label_ripeness"] != "")
        ]
        counts = ripeness_df["label_ripeness"].value_counts().to_dict()
        total = sum(counts.values())
        if total > 0:
            num_classes = len(classes_ripeness)
            weights = []
            for cls in classes_ripeness:
                cls_count = counts.get(cls, 0)
                if cls_count > 0:
                    weights.append(total / (num_classes * cls_count))
                else:
                    weights.append(1.0)
            ripeness_class_weights = torch.tensor(weights, dtype=torch.float32)

    if use_defect_pos_weight:
        defect_counts = train_dataset.df["label_defect"].value_counts().to_dict()
        pos = defect_counts.get("defective", 0)
        neg = defect_counts.get("good", 0)
        if pos > 0 and neg > 0:
            defect_pos_weight = torch.tensor(neg / pos, dtype=torch.float32)

    sampler = None
    if bool(train_cfg.get("use_sampler", True)):
        ripeness_labels = train_dataset.df["label_ripeness"].fillna("").tolist()
        if ripeness_class_weights is not None:
            weights_map = {cls: w.item() for cls, w in zip(classes_ripeness, ripeness_class_weights)}
        else:
            weights_map = {cls: 1.0 for cls in classes_ripeness}
        
        weights_map[""] = 1.0 # default for unknown
        sample_weights = [weights_map.get(label, 1.0) for label in ripeness_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        logger.info("Using WeightedRandomSampler for training")

    def train_once(worker_count: int) -> None:
        train_loader, val_loader, pin_memory = build_loaders(worker_count, sampler=sampler)
        logger.info(
            "DataLoader config | num_workers=%d | pin_memory=%s",
            worker_count,
            pin_memory,
        )

        model_name = train_cfg.get("model_name", "efficientnet_b0")
        pretrained = bool(train_cfg.get("pretrained", True))
        model = build_model(model_name, num_ripeness=len(classes_ripeness), pretrained=pretrained)
        model.to(device)

        ripeness_weights_device = (
            ripeness_class_weights.to(device) if ripeness_class_weights is not None else None
        )
        defect_pos_weight_device = (
            defect_pos_weight.to(device) if defect_pos_weight is not None else None
        )

        lr = float(train_cfg.get("lr", 3e-4))
        weight_decay = float(train_cfg.get("weight_decay", 0.01))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        epochs = int(train_cfg.get("epochs", 12))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        lambda_defect = float(train_cfg.get("lambda_defect", 1.0))
        use_amp = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"
        device_type = "cuda" if device.type == "cuda" else "cpu"
        scaler = GradScaler(device_type, enabled=use_amp)

        patience = int(train_cfg.get("patience", 5))
        best_metric_mode = train_cfg.get("best_metric", "sum_f1")

        best_metric = -1e9
        patience_counter = 0

        log_rows = []
        for epoch in range(1, epochs + 1):
            train_stats = run_epoch(
                model,
                train_loader,
                device,
                lambda_defect,
                num_ripeness=len(classes_ripeness),
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
                ripeness_class_weights=ripeness_weights_device,
                defect_pos_weight=defect_pos_weight_device,
            )
            val_stats = run_epoch(
                model,
                val_loader,
                device,
                lambda_defect,
                num_ripeness=len(classes_ripeness),
                optimizer=None,
                scaler=None,
                use_amp=use_amp,
                ripeness_class_weights=ripeness_weights_device,
                defect_pos_weight=defect_pos_weight_device,
            )

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %d | train_loss %.4f | val_loss %.4f | val_defect_f1 %.4f | val_ripeness_f1 %.4f",
                epoch,
                train_stats["loss"],
                val_stats["loss"],
                val_stats["defect_f1"],
                val_stats["ripeness_f1"],
            )
            print(
                f"Epoch {epoch}/{epochs} | train_loss {train_stats['loss']:.4f} | val_loss {val_stats['loss']:.4f} "
                f"| train_ripeness_acc {train_stats['ripeness_acc']:.4f} | train_defect_f1 {train_stats['defect_f1']:.4f} "
                f"| val_ripeness_acc {val_stats['ripeness_acc']:.4f} | val_defect_f1 {val_stats['defect_f1']:.4f}"
            )

            log_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_stats["loss"],
                    "val_loss": val_stats["loss"],
                    "train_ripeness_acc": train_stats["ripeness_acc"],
                    "train_ripeness_f1": train_stats["ripeness_f1"],
                    "train_defect_acc": train_stats["defect_acc"],
                    "train_defect_f1": train_stats["defect_f1"],
                    "val_ripeness_acc": val_stats["ripeness_acc"],
                    "val_ripeness_f1": val_stats["ripeness_f1"],
                    "val_defect_acc": val_stats["defect_acc"],
                    "val_defect_f1": val_stats["defect_f1"],
                    "lr": current_lr,
                }
            )

            checkpoint_dir = ROOT / "models" / "checkpoints"
            save_checkpoint(
                checkpoint_dir / "last.pth",
                model,
                optimizer,
                epoch,
                best_metric,
                train_cfg,
                data_cfg,
            )

            if best_metric_mode == "defect_f1":
                metric_value = val_stats["defect_f1"]
            else:
                metric_value = val_stats["defect_f1"] + val_stats["ripeness_f1"]

            if metric_value > best_metric:
                best_metric = metric_value
                patience_counter = 0
                save_checkpoint(
                    checkpoint_dir / "best.pth",
                    model,
                    optimizer,
                    epoch,
                    best_metric,
                    train_cfg,
                    data_cfg,
                )
                logger.info("Saved best checkpoint (metric=%.4f)", best_metric)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break

        log_path = ROOT / "reports" / "metrics" / "train_log.csv"
        ensure_dir(log_path.parent)
        with open(log_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = list(log_rows[0].keys()) if log_rows else []
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(log_rows)

        logger.info("Training log saved to %s", log_path)

    try:
        train_once(num_workers)
    except RuntimeError as exc:
        if os.name == "nt" and is_shared_memory_error(exc):
            logger.warning("Shared-memory DataLoader error detected. Retrying with num_workers=0.")
            train_once(0)
        else:
            raise


if __name__ == "__main__":
    main()
