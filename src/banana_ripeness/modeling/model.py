from __future__ import annotations

import torch.nn as nn
from torchvision import models


class MultiTaskEfficientNet(nn.Module):
    def __init__(self, backbone: nn.Module, in_features: int, num_ripeness: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.ripeness_head = nn.Linear(in_features, num_ripeness)
        self.defect_head = nn.Linear(in_features, 1)

    def forward(self, x):
        feats = self.backbone(x)
        ripeness_logits = self.ripeness_head(feats)
        defect_logits = self.defect_head(feats).squeeze(1)
        return ripeness_logits, defect_logits


def build_model(model_name: str, num_ripeness: int, pretrained: bool = True) -> nn.Module:
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
    elif model_name == "efficientnet_b1":
        weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b1(weights=weights)
    elif model_name == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b2(weights=weights)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    return MultiTaskEfficientNet(backbone, in_features, num_ripeness)
