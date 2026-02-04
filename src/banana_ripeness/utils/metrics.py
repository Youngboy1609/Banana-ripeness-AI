from __future__ import annotations

from typing import Iterable, List

import numpy as np


def accuracy_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true_arr = np.array(list(y_true))
    y_pred_arr = np.array(list(y_pred))
    if y_true_arr.size == 0:
        return 0.0
    return float((y_true_arr == y_pred_arr).mean())


def confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            cm[true_label, pred_label] += 1
    return cm


def macro_f1_score(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    f1s = []
    for idx in range(num_classes):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = cm[idx, :].sum()
        if support > 0:
            f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def f1_score_binary(y_true: Iterable[int], y_pred: Iterable[int], positive_label: int = 1) -> float:
    y_true_arr = np.array(list(y_true))
    y_pred_arr = np.array(list(y_pred))
    if y_true_arr.size == 0:
        return 0.0
    tp = np.sum((y_true_arr == positive_label) & (y_pred_arr == positive_label))
    fp = np.sum((y_true_arr != positive_label) & (y_pred_arr == positive_label))
    fn = np.sum((y_true_arr == positive_label) & (y_pred_arr != positive_label))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float((2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0)


def classification_report(y_true: Iterable[int], y_pred: Iterable[int], target_names: List[str]) -> str:
    num_classes = len(target_names)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    lines = []
    header = f"{'class':<20} {'precision':>9} {'recall':>9} {'f1':>9} {'support':>9}"
    lines.append(header)
    precisions = []
    recalls = []
    f1s = []
    supports = []
    for idx, name in enumerate(target_names):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        support = cm[idx, :].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        lines.append(f"{name:<20} {precision:9.4f} {recall:9.4f} {f1:9.4f} {support:9d}")
        if support > 0:
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(support)

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    total_support = int(np.sum(supports)) if supports else 0
    lines.append("")
    lines.append(
        f"{'macro avg':<20} {macro_precision:9.4f} {macro_recall:9.4f} {macro_f1:9.4f} {total_support:9d}"
    )
    return "\n".join(lines)
