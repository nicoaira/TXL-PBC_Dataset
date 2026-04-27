#!/usr/bin/env python
"""Fine-tune granular WBC model on micro_yolo_converted.zip — classification of WBC only.

What this does:
  * Extracts micro_yolo_converted.zip, which has 2 classes (0=RBC, 1=WBC).
  * Pseudo-labels each WBC box (class 1) with the model's current best WBC
    subtype prediction (Neutrophil/Lymphocyte/Monocyte/Eosinophil/Basophil),
    matched by IoU. RBC boxes stay as class 0. Result: a 7-class label set
    that aligns with the existing granular model. This makes "any of the 5
    WBC subtypes is OK" automatic — we just teach the model to predict
    *some* WBC subtype rather than RBC for those boxes.
  * Freezes the entire network except the detect head's classification
    branch (cv3 in YOLOv8/YOLO11 heads), and zeros the box/DFL loss
    weights so boxes, RBC, and platelet behavior are not disturbed.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from ultralytics import YOLO

ROOT = Path("/home/nicolas/Desktop/archivo-abril-26/TXL-PBC_Dataset")
ZIP_PATH = ROOT / "micro_yolo_converted.zip"
WORK_DIR = ROOT / "finetune_micro_dataset"
MODEL_PATH = ROOT / "runs/finetune/granular_wbc_ft_v2/weights/best.pt"

CLASS_NAMES = ["RBC", "Platelets", "Neutrophil", "Lymphocyte",
               "Monocyte", "Eosinophil", "Basophil"]
WBC_SUBTYPE_IDS = [2, 3, 4, 5, 6]
DEFAULT_WBC_ID = 2  # fallback Neutrophil if no model prediction overlaps the GT box

RUN_NAME = "granular_wbc_ft_v3_cls_only"


def iou_xywhn(a, b) -> float:
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def build_dataset() -> Path:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(WORK_DIR)

    src = WORK_DIR / "micro_yolo_converted" / "obj_train_data"
    img_dir = WORK_DIR / "images" / "train"
    lbl_dir = WORK_DIR / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    model = YOLO(str(MODEL_PATH))

    wbc_remapped = 0
    rbc_kept = 0
    for img in sorted(src.glob("*.png")):
        gt_path = src / f"{img.stem}.txt"
        if not gt_path.exists():
            continue
        shutil.copy(img, img_dir / img.name)

        pred = model.predict(
            str(img),
            conf=0.05, iou=0.5,
            classes=WBC_SUBTYPE_IDS,  # force WBC-subtype scoring even if RBC dominates
            imgsz=640, verbose=False,
        )[0].boxes
        pred_xywhn = pred.xywhn.cpu().tolist() if len(pred) else []
        pred_cls = pred.cls.cpu().tolist() if len(pred) else []

        out_lines = []
        for line in gt_path.read_text().strip().splitlines():
            parts = line.split()
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            if cls == 0:
                out_lines.append(f"0 {x} {y} {w} {h}")
                rbc_kept += 1
            elif cls == 1:
                best_cls, best_iou = DEFAULT_WBC_ID, 0.0
                for pb, pc in zip(pred_xywhn, pred_cls):
                    j = iou_xywhn((x, y, w, h), pb)
                    if j > best_iou:
                        best_iou, best_cls = j, int(pc)
                out_lines.append(f"{best_cls} {x} {y} {w} {h}")
                wbc_remapped += 1
                print(f"  {img.name}: WBC -> {CLASS_NAMES[best_cls]} (iou={best_iou:.2f})")

        (lbl_dir / f"{img.stem}.txt").write_text("\n".join(out_lines) + "\n")

    print(f"[dataset] RBC labels kept: {rbc_kept}, WBC remapped: {wbc_remapped}")

    data_yaml = WORK_DIR / "data.yaml"
    data_yaml.write_text(
        f"path: {WORK_DIR}\n"
        "train: images/train\n"
        "val: images/train\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    return data_yaml


def freeze_all_but_cls_head(trainer):
    """Unfreeze ONLY the detect-head classification branch (cv3)."""
    trainable = 0
    total = 0
    for name, param in trainer.model.named_parameters():
        total += param.numel()
        is_cls_branch = ".cv3." in name
        param.requires_grad = is_cls_branch
        if is_cls_branch:
            trainable += param.numel()
    print(f"[freeze] trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")


def main():
    data_yaml = build_dataset()

    model = YOLO(str(MODEL_PATH))
    model.add_callback("on_train_start", freeze_all_but_cls_head)

    model.train(
        data=str(data_yaml),
        epochs=40,
        imgsz=640,
        batch=4,
        lr0=1e-4,
        lrf=1e-2,
        cos_lr=True,
        optimizer="AdamW",
        warmup_epochs=1,

        # Loss: classification only — boxes/DFL must not move.
        box=0.0,
        dfl=0.0,
        cls=1.0,

        # Augmentations off (tiny dataset, labels are box-tight already).
        mosaic=0.0,
        mixup=0.0,
        close_mosaic=0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        fliplr=0.5, flipud=0.0,
        erasing=0.0,

        project=str(ROOT / "runs/finetune"),
        name=RUN_NAME,
        exist_ok=True,
        device=0,
        workers=2,
        patience=0,
        save=True,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
