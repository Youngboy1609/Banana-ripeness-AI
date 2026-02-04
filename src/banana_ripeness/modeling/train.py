from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from banana_ripeness.utils.metrics import accuracy_score, f1_score_binary, macro_f1_score


def compute_loss(
    logits_ripeness: torch.Tensor,
    logits_defect: torch.Tensor,
    y_ripeness: torch.Tensor,
    mask_ripeness: torch.Tensor,
    y_defect: torch.Tensor,
    lambda_defect: float,
    ripeness_class_weights: torch.Tensor | None = None,
    defect_pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    defect_loss = F.binary_cross_entropy_with_logits(
        logits_defect, y_defect, pos_weight=defect_pos_weight
    )
    if mask_ripeness.any():
        ripeness_loss = F.cross_entropy(
            logits_ripeness[mask_ripeness],
            y_ripeness[mask_ripeness],
            weight=ripeness_class_weights,
        )
    else:
        ripeness_loss = torch.tensor(0.0, device=logits_defect.device)
    total_loss = ripeness_loss + lambda_defect * defect_loss
    return total_loss, ripeness_loss, defect_loss


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    lambda_defect: float,
    num_ripeness: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
    ripeness_class_weights: torch.Tensor | None = None,
    defect_pos_weight: torch.Tensor | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_samples = 0

    ripeness_correct = 0
    ripeness_total = 0
    defect_correct = 0
    defect_total = 0
    defect_tp = 0
    defect_fp = 0
    defect_fn = 0

    y_true_ripeness = []
    y_pred_ripeness = []
    y_true_defect = []
    y_pred_defect = []

    device_type = "cuda" if device.type == "cuda" else "cpu"
    loader_len = len(loader)
    progress = tqdm(loader, total=loader_len, desc="train" if is_train else "val", leave=False)
    for step, batch in enumerate(progress):
        images, y_ripeness, mask_ripeness, y_defect = batch[:4]
        images = images.to(device, non_blocking=True)
        y_ripeness = y_ripeness.to(device, non_blocking=True)
        mask_ripeness = mask_ripeness.to(device, non_blocking=True)
        y_defect = y_defect.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device_type, enabled=use_amp):
            logits_ripeness, logits_defect = model(images)
            loss, _, _ = compute_loss(
                logits_ripeness,
                logits_defect,
                y_ripeness,
                mask_ripeness,
                y_defect,
                lambda_defect,
                ripeness_class_weights=ripeness_class_weights,
                defect_pos_weight=defect_pos_weight,
            )

        if is_train:
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        ripeness_pred = torch.argmax(logits_ripeness, dim=1)
        defect_prob = torch.sigmoid(logits_defect)
        defect_pred = (defect_prob >= 0.5).long()

        if mask_ripeness.any():
            masked_true = y_ripeness[mask_ripeness]
            masked_pred = ripeness_pred[mask_ripeness]
            ripeness_correct += (masked_pred == masked_true).sum().item()
            ripeness_total += masked_true.numel()
            y_true_ripeness.extend(y_ripeness[mask_ripeness].cpu().tolist())
            y_pred_ripeness.extend(ripeness_pred[mask_ripeness].cpu().tolist())

        defect_true = y_defect.long()
        defect_correct += (defect_pred == defect_true).sum().item()
        defect_total += defect_true.numel()
        defect_tp += ((defect_true == 1) & (defect_pred == 1)).sum().item()
        defect_fp += ((defect_true == 0) & (defect_pred == 1)).sum().item()
        defect_fn += ((defect_true == 1) & (defect_pred == 0)).sum().item()

        y_true_defect.extend(y_defect.long().cpu().tolist())
        y_pred_defect.extend(defect_pred.cpu().tolist())

        if (step + 1) % 50 == 0 or (step + 1) == loader_len:
            running_loss = total_loss / max(1, total_samples)
            running_ripeness_acc = ripeness_correct / ripeness_total if ripeness_total > 0 else 0.0
            running_defect_acc = defect_correct / defect_total if defect_total > 0 else 0.0
            denom = (2 * defect_tp) + defect_fp + defect_fn
            running_defect_f1 = (2 * defect_tp / denom) if denom > 0 else 0.0
            progress.set_postfix(
                {
                    "loss": f"{running_loss:.4f}",
                    "ripeness_acc": f"{running_ripeness_acc:.4f}",
                    "defect_acc": f"{running_defect_acc:.4f}",
                    "defect_f1": f"{running_defect_f1:.4f}",
                }
            )

    avg_loss = total_loss / max(1, total_samples)
    ripeness_acc = accuracy_score(y_true_ripeness, y_pred_ripeness)
    ripeness_f1 = macro_f1_score(y_true_ripeness, y_pred_ripeness, num_ripeness)
    defect_acc = accuracy_score(y_true_defect, y_pred_defect)
    defect_f1 = f1_score_binary(y_true_defect, y_pred_defect, positive_label=1)

    return {
        "loss": avg_loss,
        "ripeness_acc": ripeness_acc,
        "ripeness_f1": ripeness_f1,
        "defect_acc": defect_acc,
        "defect_f1": defect_f1,
    }
