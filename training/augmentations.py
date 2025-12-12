"""Albumentations pipeline for detection training.

This module centralizes strong augmentations (RandomCrop, MotionBlur,
ColorJitter, CutOut/CoarseDropout, light affine) so we keep the training
loop clean. Bounding boxes are assumed to be in YOLO format
``[x_center, y_center, width, height]`` normalized to image size.
"""

from __future__ import annotations

from typing import Tuple

import albumentations as A


def build_detection_augmentations(
    img_size: int = 640,
    crop_scale: Tuple[float, float] = (0.8, 1.0),
    crop_ratio: Tuple[float, float] = (0.85, 1.15),
    p: float = 1.0,
) -> A.Compose:
    """
    Slightly lighter Albumentations pipeline to stay GPU-friendly in Colab.
    """
    rrc = A.RandomResizedCrop(size=(img_size, img_size), scale=crop_scale, ratio=crop_ratio, p=0.25)

    try:
        dropout = A.Cutout(
            num_holes=3,
            max_h_size=int(img_size * 0.08),
            max_w_size=int(img_size * 0.08),
            fill_value=114,
            p=0.2,
        )
    except Exception:
        dropout = A.NoOp()

    return A.Compose(
        [
            rrc,
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(0.02, 0.06),
                rotate=(-3, 3),
                shear=(-1.5, 1.5),
                p=0.25,
            ),
            A.Perspective(scale=(0.015, 0.035), p=0.15),
            A.MotionBlur(blur_limit=5, p=0.1),
            A.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.05, p=0.25),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.15),
            A.HorizontalFlip(p=0.5),
            dropout,
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.1,
            min_area=4,
        ),
        p=p,
    )
