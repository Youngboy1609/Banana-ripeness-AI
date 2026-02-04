from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banana_ripeness.modeling.model import build_model
from banana_ripeness.utils.io import ensure_dir, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", default="models/checkpoints/best.pth", help="Checkpoint path")
    parser.add_argument("--output", default="models/exported/model.onnx", help="Output ONNX path")
    parser.add_argument("--data-config", default=None, help="Optional data config for labels")
    parser.add_argument("--img-size", type=int, default=None, help="Override image size")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes_ripeness = checkpoint.get("classes_ripeness")
    classes_defect = checkpoint.get("classes_defect")

    if (classes_ripeness is None or classes_defect is None) and args.data_config:
        data_cfg_path = Path(args.data_config)
        if not data_cfg_path.is_absolute():
            data_cfg_path = ROOT / data_cfg_path
        data_cfg = load_yaml(data_cfg_path)
        classes_ripeness = data_cfg.get("classes_ripeness", [])
        classes_defect = data_cfg.get("classes_defect", [])

    if classes_ripeness is None:
        raise RuntimeError("Missing classes_ripeness in checkpoint and no data config provided.")
    if classes_defect is None:
        raise RuntimeError("Missing classes_defect in checkpoint and no data config provided.")

    model_name = checkpoint.get("model_name", "efficientnet_b0")
    model = build_model(model_name, num_ripeness=len(classes_ripeness), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    img_size = args.img_size or checkpoint.get("img_size", 224)
    dummy_input = torch.randn(1, 3, img_size, img_size)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    ensure_dir(output_path.parent)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["ripeness_logits", "defect_logits"],
        opset_version=12,
    )

    labels_path = output_path.parent / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"classes_ripeness": classes_ripeness, "classes_defect": classes_defect},
            handle,
            indent=2,
        )

    print(f"[export] Saved ONNX to {output_path}")
    print(f"[export] Saved labels to {labels_path}")


if __name__ == "__main__":
    main()
