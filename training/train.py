import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on the COCO person subset"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset.yaml",
        help="Path to YOLO dataset.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Base model checkpoint (e.g., yolov8s.pt, yolov8s.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (T4-friendly default)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (T4-friendly default)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='Device id, e.g. "0", "0,1", or "cpu"',
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Dataloader workers",
    )
    parser.add_argument(
        "--multi_scale",
        action="store_true",
        default=True,
        help="Enable multi-scale training",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Enable strong augmentations",
    )
    parser.add_argument(
        "--use_albumentations",
        action="store_true",
        default=False,
        help="Use custom Albumentations pipeline (see augmentations.py)",
    )
    parser.add_argument(
        "--albu_prob",
        type=float,
        default=0.7,
        help="Global probability for the Albumentations pipeline",
    )
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.1,
        help="MixUp probability",
    )
    parser.add_argument(
        "--copy_paste",
        type=float,
        default=0.1,
        help="Copy-paste probability (proxy for cutout/patch augment)",
    )
    parser.add_argument(
        "--erasing",
        type=float,
        default=0.4,
        help="Random erasing probability (cutout-style)",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parent / "logs",
        help="Base directory for training outputs (Plan: training/logs)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov8-person",
        help="Run name under project/",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the last run in project/name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="cv-advanced-assessment",
        help="W&B project name (offline by default)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (optional)",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="offline",
        help="W&B mode; set to 'offline' if no internet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.wandb_project:
        import os

        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_mode:
            os.environ["WANDB_MODE"] = args.wandb_mode
    data_path = str(args.data)
    project_dir = Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    # Optional: custom Albumentations pipeline on top of Ultralytics defaults
    if args.use_albumentations:
        try:
            training_dir = Path(__file__).resolve().parent
            if str(training_dir) not in sys.path:
                sys.path.insert(0, str(training_dir))
            import numpy as np
            import torch
            from augmentations import build_detection_augmentations
        except Exception as exc:  # pragma: no cover - runtime import guard
            raise RuntimeError("Albumentations pipeline requested but not available") from exc

        albu = build_detection_augmentations(img_size=args.imgsz, p=args.albu_prob)

        def _apply_albu(trainer) -> None:
            """
            Callback runs before each train batch to apply Albumentations.

            Ultralytics batches provide:
            - batch["img"]: Tensor [B, 3, H, W]
            - batch["bboxes"]: Tensor [N, 4] (YOLO normalized)
            - batch["cls"]: Tensor [N, 1] or [N]
            - batch["batch_idx"]: Tensor [N] mapping each bbox to its image
            """

            batch = getattr(trainer, "batch", None)
            if not isinstance(batch, dict):
                return
            imgs = batch.get("img")
            bboxes = batch.get("bboxes")
            classes = batch.get("cls")
            batch_idx = batch.get("batch_idx")
            if imgs is None or bboxes is None or classes is None or batch_idx is None:
                return

            device = imgs.device
            imgs_np = imgs.permute(0, 2, 3, 1).cpu().numpy()
            bboxes_np = bboxes.cpu().numpy()
            classes_np = classes.cpu().numpy().reshape(-1)
            batch_idx_np = batch_idx.cpu().numpy()

            new_imgs = []
            new_bboxes = []
            new_classes = []
            new_batch_idx = []

            for i, img_np in enumerate(imgs_np):
                mask = batch_idx_np == i
                boxes_i = bboxes_np[mask]
                classes_i = classes_np[mask]
                img_uint8 = np.clip(img_np, 0, 255).astype("uint8")

                if boxes_i.size == 0:
                    new_imgs.append(torch.from_numpy(img_uint8).permute(2, 0, 1))
                    continue

                transformed = albu(
                    image=img_uint8,
                    bboxes=boxes_i.tolist(),
                    class_labels=classes_i.tolist(),
                )

                new_imgs.append(torch.from_numpy(transformed["image"]).permute(2, 0, 1))

                tb = torch.tensor(transformed["bboxes"], dtype=torch.float32)
                tc = torch.tensor(transformed["class_labels"], dtype=torch.float32).view(-1, 1)
                if tb.numel():
                    new_bboxes.append(tb)
                    new_classes.append(tc)
                    new_batch_idx.append(torch.full((tb.shape[0],), i, dtype=torch.int64))

            batch["img"] = torch.stack(new_imgs).to(device, non_blocking=True)
            if new_bboxes:
                batch["bboxes"] = torch.cat(new_bboxes).to(device)
                batch["cls"] = torch.cat(new_classes).to(device)
                batch["batch_idx"] = torch.cat(new_batch_idx).to(device)
            else:
                # If augmentations drop all labels, keep tensors consistent
                batch["bboxes"] = torch.zeros((0, 4), device=device)
                batch["cls"] = torch.zeros((0, 1), device=device)
                batch["batch_idx"] = torch.zeros((0,), device=device, dtype=torch.int64)

        model.add_callback("on_train_batch_start", _apply_albu)

    train_kwargs = dict(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_dir),
        name=args.name,
        resume=args.resume,
        seed=args.seed,
        cos_lr=True,
        amp=True,
        multi_scale=args.multi_scale,
        augment=args.augment,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
        exist_ok=True,
    )
    model.train(**train_kwargs)

    # Copy best weight to models/latest.pt to align with Project Plan
    weights_dir = project_dir / args.name / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / "latest.pt"

    src = best if best.exists() else last if last.exists() else None
    if src is None:
        print(f"[warn] No weights found in {weights_dir}")
        return

    shutil.copy2(src, target)
    print(f"[info] Copied {src.name} -> {target}")


if __name__ == "__main__":
    main()

