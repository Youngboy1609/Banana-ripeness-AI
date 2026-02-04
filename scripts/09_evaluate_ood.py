from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banana_ripeness.data.dataset import MultiTaskDataset
from banana_ripeness.data.prepare import canonicalize_class_name
from banana_ripeness.inference.preprocess import build_transforms
from banana_ripeness.modeling.evaluate import collect_predictions, plot_confusion_matrix
from banana_ripeness.modeling.model import build_model
from banana_ripeness.utils.io import ensure_dir, load_yaml, write_csv, write_text
from banana_ripeness.utils.metrics import classification_report, confusion_matrix

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def build_ood_metadata(root: Path, classes_ripeness: list[str], classes_defect: list[str]) -> tuple[list[dict], int]:
    rows = []
    skipped = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        parts = rel.parts
        if len(parts) < 2:
            skipped += 1
            continue
        ripeness = canonicalize_class_name(parts[0])
        defect = canonicalize_class_name(parts[1])
        if ripeness not in classes_ripeness:
            ripeness = ""
        if defect not in classes_defect:
            skipped += 1
            continue

        try:
            path_str = path.relative_to(ROOT).as_posix()
        except ValueError:
            path_str = str(path)

        rows.append(
            {
                "path": path_str,
                "split": "test",
                "source": root.name,
                "label_ripeness": ripeness,
                "label_defect": defect,
            }
        )
    return rows, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on labeled OOD data")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to train config")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override")
    parser.add_argument("--root", default="data/webcam_captures", help="OOD root (ripeness/defect folders)")
    parser.add_argument("--output-dir", default="reports/ood", help="Output directory")
    args = parser.parse_args()

    train_cfg_path = Path(args.config)
    if not train_cfg_path.is_absolute():
        train_cfg_path = ROOT / train_cfg_path
    train_cfg = load_yaml(train_cfg_path)

    data_cfg_path = Path(train_cfg.get("data_config", "configs/data.yaml"))
    if not data_cfg_path.is_absolute():
        data_cfg_path = ROOT / data_cfg_path
    data_cfg = load_yaml(data_cfg_path)

    classes_ripeness = data_cfg.get("classes_ripeness", [])
    classes_defect = data_cfg.get("classes_defect", [])

    ood_root = Path(args.root)
    if not ood_root.is_absolute():
        ood_root = ROOT / ood_root
    if not ood_root.exists():
        raise FileNotFoundError(f"OOD root not found: {ood_root}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)

    rows, skipped = build_ood_metadata(ood_root, classes_ripeness, classes_defect)
    if not rows:
        raise RuntimeError("No labeled OOD samples found. Expected <ripeness>/<defect> folders.")

    metadata_path = metrics_dir / "ood_metadata.csv"
    write_csv(metadata_path, rows, ["path", "split", "source", "label_ripeness", "label_defect"])
    if skipped:
        print(f"[ood] Skipped {skipped} files missing labels")

    img_size = int(train_cfg.get("img_size", 224))
    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 0))
    if os.name == "nt":
        num_workers = 0 if num_workers <= 0 else 1

    test_tf = build_transforms(img_size, is_train=False)
    test_dataset = MultiTaskDataset(
        metadata_path,
        root_dir=ROOT,
        split="test",
        classes_ripeness=classes_ripeness,
        classes_defect=classes_defect,
        transform=test_tf,
        return_path=True,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    test_loader = DataLoader(test_dataset, **loader_kwargs)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else ROOT / "models" / "checkpoints" / "best.pth"
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes_ripeness = checkpoint.get("classes_ripeness", classes_ripeness)
    classes_defect = checkpoint.get("classes_defect", classes_defect)
    model_name = checkpoint.get("model_name", train_cfg.get("model_name", "efficientnet_b0"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_ripeness=len(classes_ripeness), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    use_amp = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"
    preds = collect_predictions(model, test_loader, device, use_amp=use_amp)

    mask = np.array(preds["mask_ripeness"], dtype=bool)
    y_true_ripeness = np.array(preds["y_ripeness"])[mask]
    y_pred_ripeness = np.array(preds["pred_ripeness"])[mask]

    if y_true_ripeness.size > 0:
        ripeness_report = classification_report(y_true_ripeness, y_pred_ripeness, classes_ripeness)
        cm_ripeness = confusion_matrix(y_true_ripeness, y_pred_ripeness, len(classes_ripeness))
    else:
        ripeness_report = "No ripeness labels available in OOD set."
        cm_ripeness = np.zeros((len(classes_ripeness), len(classes_ripeness)), dtype=np.int64)

    y_true_defect = preds["y_defect"]
    y_pred_defect = preds["pred_defect"]
    defect_report = classification_report(y_true_defect, y_pred_defect, classes_defect)
    cm_defect = confusion_matrix(y_true_defect, y_pred_defect, len(classes_defect))

    write_text(metrics_dir / "ripeness_classification_report.txt", ripeness_report)
    write_text(metrics_dir / "defect_classification_report.txt", defect_report)

    plot_confusion_matrix(
        cm_ripeness,
        classes_ripeness,
        figures_dir / "ripeness_confusion_matrix.png",
        "OOD Ripeness Confusion Matrix",
    )
    plot_confusion_matrix(
        cm_defect,
        classes_defect,
        figures_dir / "defect_confusion_matrix.png",
        "OOD Defect Confusion Matrix",
    )

    preds_path = metrics_dir / "preds.csv"
    with open(preds_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "true_ripeness",
                "pred_ripeness",
                "true_defect",
                "pred_defect",
                "prob_defect",
            ],
        )
        writer.writeheader()
        for idx, path_str in enumerate(preds["paths"]):
            path = Path(path_str)
            try:
                path_str = path.relative_to(ROOT).as_posix()
            except ValueError:
                path_str = str(path)

            true_r = preds["y_ripeness"][idx]
            mask_r = preds["mask_ripeness"][idx]
            pred_r = preds["pred_ripeness"][idx]

            true_r_label = classes_ripeness[true_r] if mask_r and true_r >= 0 else ""
            pred_r_label = classes_ripeness[pred_r] if mask_r and pred_r >= 0 else ""

            true_d_label = classes_defect[int(preds["y_defect"][idx])]
            pred_d_label = classes_defect[int(preds["pred_defect"][idx])]

            writer.writerow(
                {
                    "path": path_str,
                    "true_ripeness": true_r_label,
                    "pred_ripeness": pred_r_label,
                    "true_defect": true_d_label,
                    "pred_defect": pred_d_label,
                    "prob_defect": f"{preds['prob_defect'][idx]:.4f}",
                }
            )

    print(f"[ood] Wrote OOD reports to {metrics_dir}")


if __name__ == "__main__":
    main()
