from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banana_ripeness.inference.preprocess import build_transforms
from banana_ripeness.inference.webcam import MovingAverageSmoother, center_crop_roi
from banana_ripeness.modeling.model import build_model
from banana_ripeness.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam demo")
    parser.add_argument("--config", default="configs/infer.yaml", help="Path to infer config")
    args = parser.parse_args()

    infer_cfg_path = Path(args.config)
    if not infer_cfg_path.is_absolute():
        infer_cfg_path = ROOT / infer_cfg_path
    infer_cfg = load_yaml(infer_cfg_path)

    data_cfg_path = Path(infer_cfg.get("data_config", "configs/data.yaml"))
    if not data_cfg_path.is_absolute():
        data_cfg_path = ROOT / data_cfg_path
    data_cfg = load_yaml(data_cfg_path)

    classes_ripeness = data_cfg.get("classes_ripeness", [])

    checkpoint_path = Path(infer_cfg.get("checkpoint_path", "models/checkpoints/best.pth"))
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_name = checkpoint.get("model_name", "efficientnet_b0")
    classes_ripeness = checkpoint.get("classes_ripeness", classes_ripeness)

    img_size = int(infer_cfg.get("img_size", checkpoint.get("img_size", 224)))

    model = build_model(model_name, num_ripeness=len(classes_ripeness), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transform = build_transforms(img_size, is_train=False)

    roi_scale = float(infer_cfg.get("roi_scale", 0.6))
    smoothing_window = int(infer_cfg.get("smoothing_window", 8))
    conf_threshold_defect = float(infer_cfg.get("conf_threshold_defect", 0.6))
    conf_threshold_ripeness = float(infer_cfg.get("conf_threshold_ripeness", 0.5))
    camera_index = int(infer_cfg.get("camera_index", 0))

    smoother_ripeness = MovingAverageSmoother(smoothing_window)
    smoother_defect = MovingAverageSmoother(smoothing_window)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[demo] Failed to open webcam")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi, (x1, y1, x2, y2) = center_crop_roi(frame, roi_scale)
        if roi.size == 0:
            continue

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(roi_rgb)
        tensor = transform(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_ripeness, logits_defect = model(tensor)
            prob_ripeness = torch.softmax(logits_ripeness, dim=1).cpu().numpy()[0]
            prob_defect = torch.sigmoid(logits_defect).cpu().numpy()[0]

        prob_ripeness = smoother_ripeness.update(prob_ripeness)
        prob_defect = smoother_defect.update(np.array([prob_defect]))[0]

        ripeness_idx = int(np.argmax(prob_ripeness))
        ripeness_conf = float(prob_ripeness[ripeness_idx])
        if ripeness_conf >= conf_threshold_ripeness:
            ripeness_label = classes_ripeness[ripeness_idx]
        else:
            ripeness_label = "unknown"

        defect_label = "defective" if prob_defect >= conf_threshold_defect else "good"
        defect_color = (0, 0, 255) if defect_label == "defective" else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), defect_color, 2)

        cv2.putText(
            frame,
            f"Ripeness: {ripeness_label} ({ripeness_conf:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Defect: {defect_label} ({prob_defect:.2f})",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            defect_color,
            2,
        )

        now = time.time()
        fps = 1.0 / max(1e-6, now - prev_time)
        prev_time = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Banana Ripeness Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
