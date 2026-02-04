from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banana_ripeness.inference.webcam import center_crop_roi
from banana_ripeness.utils.io import ensure_dir, load_yaml


def load_data_config(config_path: Path) -> dict:
    cfg = load_yaml(config_path)
    if "classes_ripeness" in cfg:
        return cfg
    data_cfg_path = Path(cfg.get("data_config", "configs/data.yaml"))
    if not data_cfg_path.is_absolute():
        data_cfg_path = ROOT / data_cfg_path
    return load_yaml(data_cfg_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect webcam data")
    parser.add_argument("--config", default="configs/infer.yaml", help="Infer or data config")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = load_yaml(config_path)
    data_cfg = load_data_config(config_path)

    classes_ripeness = data_cfg.get("classes_ripeness", [])
    classes_defect = data_cfg.get("classes_defect", ["good", "defective"])

    camera_index = int(cfg.get("camera_index", 0))
    roi_scale = float(cfg.get("roi_scale", 0.6))

    current_ripeness_idx = 0
    current_defect = classes_defect[0]
    total_saved = 0
    counts = {label: {defect: 0 for defect in classes_defect} for label in classes_ripeness}

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[collect] Failed to open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi, (x1, y1, x2, y2) = center_crop_roi(frame, roi_scale)
        if roi.size == 0:
            continue

        ripeness_label = classes_ripeness[current_ripeness_idx] if classes_ripeness else "unknown"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Label: {ripeness_label} | Defect: {current_defect}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "Keys: 1-4 ripeness, d toggle defect, s save, q quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Saved: {total_saved}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        if classes_ripeness:
            y_offset = 120
            for label in classes_ripeness:
                good_count = counts[label].get(classes_defect[0], 0)
                bad_count = counts[label].get(classes_defect[-1], 0) if len(classes_defect) > 1 else 0
                cv2.putText(
                    frame,
                    f"{label}: {classes_defect[0]}={good_count} {classes_defect[-1]}={bad_count}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    2,
                )
                y_offset += 22

        cv2.imshow("Collect Webcam Data", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("1"), ord("2"), ord("3"), ord("4")) and classes_ripeness:
            idx = int(chr(key)) - 1
            if idx < len(classes_ripeness):
                current_ripeness_idx = idx
        elif key in (ord("d"), ord("D")):
            current_defect = classes_defect[1] if current_defect == classes_defect[0] else classes_defect[0]
        elif key in (ord("s"), ord("S")):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_dir = ROOT / "data" / "webcam_captures" / ripeness_label / current_defect
            ensure_dir(save_dir)
            save_path = save_dir / f"{timestamp}.jpg"
            cv2.imwrite(str(save_path), roi)
            total_saved += 1
            if ripeness_label in counts and current_defect in counts[ripeness_label]:
                counts[ripeness_label][current_defect] += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
