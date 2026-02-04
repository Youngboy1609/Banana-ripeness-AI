from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banana_ripeness.data.dataset import MultiTaskDataset
from banana_ripeness.inference.preprocess import build_transforms
from banana_ripeness.modeling.evaluate import collect_predictions, plot_confusion_matrix
from banana_ripeness.modeling.model import build_model
from banana_ripeness.utils.io import ensure_dir, load_yaml, write_text
from banana_ripeness.utils.metrics import classification_report, confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-task model")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to train config")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override")
    parser.add_argument("--metadata", default=None, help="Override metadata CSV path")
    parser.add_argument("--output-dir", default=None, help="Output directory for metrics/figures")
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

    if args.metadata:
        metadata_path = Path(args.metadata)
        if not metadata_path.is_absolute():
            metadata_path = ROOT / metadata_path
    else:
        metadata_path = ROOT / "data" / "processed" / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}. Run 02_prepare_dataset.py first.")

    img_size = int(train_cfg.get("img_size", 224))
    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 4))

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else ROOT / "models" / "checkpoints" / "best.pth"
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes_ripeness = checkpoint.get("classes_ripeness", classes_ripeness)
    classes_defect = checkpoint.get("classes_defect", classes_defect)
    model_name = checkpoint.get("model_name", train_cfg.get("model_name", "efficientnet_b0"))
    model = build_model(model_name, num_ripeness=len(classes_ripeness), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    use_amp = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"
    preds = collect_predictions(model, test_loader, device, use_amp=use_amp)

    path_to_source = {}
    for _, row in test_dataset.df.iterrows():
        path_to_source[str(row["path"])] = row["source"]

    mask = np.array(preds["mask_ripeness"], dtype=bool)
    y_true_ripeness = np.array(preds["y_ripeness"])[mask]
    y_pred_ripeness = np.array(preds["pred_ripeness"])[mask]

    if y_true_ripeness.size > 0:
        ripeness_report = classification_report(y_true_ripeness, y_pred_ripeness, classes_ripeness)
        cm_ripeness = confusion_matrix(y_true_ripeness, y_pred_ripeness, len(classes_ripeness))
    else:
        ripeness_report = "No ripeness labels available in test set."
        cm_ripeness = np.zeros((len(classes_ripeness), len(classes_ripeness)), dtype=np.int64)

    y_true_defect = preds["y_defect"]
    y_pred_defect = preds["pred_defect"]
    defect_report = classification_report(y_true_defect, y_pred_defect, classes_defect)
    cm_defect = confusion_matrix(y_true_defect, y_pred_defect, len(classes_defect))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        metrics_dir = output_dir / "metrics"
        figures_dir = output_dir / "figures"
    else:
        metrics_dir = ROOT / "reports" / "metrics"
        figures_dir = ROOT / "reports" / "figures"
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)

    write_text(metrics_dir / "ripeness_classification_report.txt", ripeness_report)
    write_text(metrics_dir / "defect_classification_report.txt", defect_report)

    plot_confusion_matrix(
        cm_ripeness,
        classes_ripeness,
        figures_dir / "ripeness_confusion_matrix.png",
        "Ripeness Confusion Matrix",
    )
    plot_confusion_matrix(
        cm_defect,
        classes_defect,
        figures_dir / "defect_confusion_matrix.png",
        "Defect Confusion Matrix",
    )

    preds_path = metrics_dir / "preds.csv"
    per_source = {}
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

            source = path_to_source.get(path_str, "unknown")
            if source not in per_source:
                per_source[source] = {
                    "ripeness_true": [],
                    "ripeness_pred": [],
                    "defect_true": [],
                    "defect_pred": [],
                }

            true_r = preds["y_ripeness"][idx]
            mask_r = preds["mask_ripeness"][idx]
            pred_r = preds["pred_ripeness"][idx]

            true_r_label = classes_ripeness[true_r] if mask_r and true_r >= 0 else ""
            pred_r_label = classes_ripeness[pred_r] if mask_r and pred_r >= 0 else ""

            true_d_label = classes_defect[int(preds["y_defect"][idx])]
            pred_d_label = classes_defect[int(preds["pred_defect"][idx])]

            if mask_r:
                per_source[source]["ripeness_true"].append(true_r)
                per_source[source]["ripeness_pred"].append(pred_r)
            per_source[source]["defect_true"].append(int(preds["y_defect"][idx]))
            per_source[source]["defect_pred"].append(int(preds["pred_defect"][idx]))

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

    per_source_lines = []
    for source, data in sorted(per_source.items()):
        per_source_lines.append(f"Source: {source}")
        per_source_lines.append(
            f"  ripeness samples: {len(data['ripeness_true'])} | defect samples: {len(data['defect_true'])}"
        )
        if data["ripeness_true"]:
            report = classification_report(
                data["ripeness_true"], data["ripeness_pred"], classes_ripeness
            )
        else:
            report = "No ripeness labels available."
        per_source_lines.append("  Ripeness report:")
        per_source_lines.extend([f"  {line}" for line in report.splitlines()])

        defect_report_src = classification_report(
            data["defect_true"], data["defect_pred"], classes_defect
        )
        per_source_lines.append("  Defect report:")
        per_source_lines.extend([f"  {line}" for line in defect_report_src.splitlines()])
        per_source_lines.append("")

    write_text(metrics_dir / "per_source_reports.txt", "\n".join(per_source_lines).rstrip())

    print(f"[evaluate] Wrote reports to {metrics_dir}")


if __name__ == "__main__":
    main()
