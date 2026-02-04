from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int, is_train: bool, augment_cfg: dict | None = None):
    if is_train:
        cfg = augment_cfg or {}
        crop_cfg = cfg.get("random_resized_crop", {}) or {}
        scale_min = float(crop_cfg.get("scale_min", 0.7))
        scale_max = float(crop_cfg.get("scale_max", 1.0))

        aug = [
            transforms.RandomResizedCrop(img_size, scale=(scale_min, scale_max)),
        ]

        if cfg.get("hflip", True):
            aug.append(transforms.RandomHorizontalFlip())

        rotation = float(cfg.get("rotation", 0.0))
        if rotation > 0:
            aug.append(transforms.RandomRotation(rotation))

        perspective_prob = float(cfg.get("perspective_prob", 0.0))
        if perspective_prob > 0:
            aug.append(transforms.RandomPerspective(distortion_scale=0.3, p=perspective_prob))

        color_cfg = cfg.get("color_jitter", {}) or {}
        brightness = float(color_cfg.get("brightness", 0.0))
        contrast = float(color_cfg.get("contrast", 0.0))
        saturation = float(color_cfg.get("saturation", 0.0))
        hue = float(color_cfg.get("hue", 0.0))
        if any(val > 0 for val in (brightness, contrast, saturation, hue)):
            aug.append(
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            )

        grayscale_prob = float(cfg.get("grayscale_prob", 0.0))
        if grayscale_prob > 0:
            aug.append(transforms.RandomGrayscale(p=grayscale_prob))

        blur_prob = float(cfg.get("blur_prob", 0.0))
        if blur_prob > 0:
            aug.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                    p=blur_prob,
                )
            )

        aug.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        erasing_prob = float(cfg.get("random_erasing_prob", 0.0))
        if erasing_prob > 0:
            scale_min = float(cfg.get("random_erasing_scale_min", 0.02))
            scale_max = float(cfg.get("random_erasing_scale_max", 0.15))
            aug.append(
                transforms.RandomErasing(
                    p=erasing_prob,
                    scale=(scale_min, scale_max),
                    ratio=(0.3, 3.3),
                    value="random",
                )
            )

        return transforms.Compose(aug)
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
