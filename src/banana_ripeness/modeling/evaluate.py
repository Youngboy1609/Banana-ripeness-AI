from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast


def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, List]:
    model.eval()
    paths: List[str] = []
    y_ripeness: List[int] = []
    mask_ripeness: List[int] = []
    y_defect: List[int] = []
    pred_ripeness: List[int] = []
    pred_defect: List[int] = []
    prob_defect: List[float] = []

    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad():
        for batch in loader:
            images, y_ripeness_b, mask_ripeness_b, y_defect_b, paths_b = batch
            images = images.to(device, non_blocking=True)

            with autocast(device_type=device_type, enabled=use_amp):
                logits_ripeness, logits_defect = model(images)

            ripeness_pred_b = torch.argmax(logits_ripeness, dim=1)
            defect_prob_b = torch.sigmoid(logits_defect)
            defect_pred_b = (defect_prob_b >= 0.5).long()

            paths.extend(paths_b)
            y_ripeness.extend(y_ripeness_b.cpu().tolist())
            mask_ripeness.extend(mask_ripeness_b.cpu().int().tolist())
            y_defect.extend(y_defect_b.long().cpu().tolist())
            pred_ripeness.extend(ripeness_pred_b.cpu().tolist())
            pred_defect.extend(defect_pred_b.cpu().tolist())
            prob_defect.extend(defect_prob_b.cpu().tolist())

    return {
        "paths": paths,
        "y_ripeness": y_ripeness,
        "mask_ripeness": mask_ripeness,
        "y_defect": y_defect,
        "pred_ripeness": pred_ripeness,
        "pred_defect": pred_defect,
        "prob_defect": prob_defect,
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
