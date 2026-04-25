"""Box-level confusion matrix for the granular WBC model.

Ground truth comes from wbc_reviewed/ labels (CVAT export), which already
contain RBC, WBC, and Platelet boxes. The single 'WBC' class is split into
its specific subclass using the image filename.

Each ground-truth box is matched to the best-overlapping predicted box
(IoU >= IOU_MATCH). If no prediction overlaps, it counts as 'No detection'.
False positives (predicted boxes with no GT match) are summarised separately.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR   = PROJECT_ROOT / "wbc_reviewed" / "obj_train_data" / "obj_train_data"
CHECKPOINT   = PROJECT_ROOT / "runs" / "finetune" / "granular_wbc_ft" / "weights" / "best.pt"

WBC_SUBCLASSES = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]
TRUE_CLASSES   = ["RBC", "Platelets"] + WBC_SUBCLASSES

# CVAT class IDs in wbc_reviewed: 0=RBC, 1=WBC, 2=Platelets
CVAT_FIXED = {0: "RBC", 2: "Platelets"}

IOU_MATCH = 0.3


def wbc_subclass_from_stem(stem: str):
    parts = stem.split("_")
    if len(parts) >= 2 and parts[1] in WBC_SUBCLASSES:
        return parts[1]
    return None


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / (a_area + b_area - inter)


def parse_gt(label_path: Path, img_w: int, img_h: int, wbc_sub: str):
    """Return list of (true_class_name, xyxy)."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cvat_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])

        if cvat_id == 1:           # CVAT WBC → specific subclass
            cls = wbc_sub
        else:
            cls = CVAT_FIXED.get(cvat_id)
        if cls is None:
            continue

        boxes.append((cls, yolo_to_xyxy(cx, cy, w, h, img_w, img_h)))
    return boxes


def run(checkpoint: Path, source_dir: Path, conf: float, iou_match: float, imgsz: int):
    try:
        from ultralytics import YOLO
        from PIL import Image
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(f"Missing dep: {e}")

    print(f"Loading model: {checkpoint}")
    model = YOLO(str(checkpoint))
    model_names = [model.names[i] for i in sorted(model.names)]
    pred_labels = model_names + ["No detection"]
    print(f"Model classes: {model_names}\n")

    images = sorted(p for p in source_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    print(f"Found {len(images)} images\n")

    counts = {t: defaultdict(int) for t in TRUE_CLASSES}
    false_positives = defaultdict(int)   # predicted class with no GT match
    skipped = 0

    for img_path in images:
        wbc_sub = wbc_subclass_from_stem(img_path.stem)
        if wbc_sub is None:
            skipped += 1
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        gt_boxes = parse_gt(img_path.with_suffix(".txt"), img_w, img_h, wbc_sub)
        if not gt_boxes:
            continue

        results = model.predict(
            source=str(img_path), conf=conf,
            iou=0.7, imgsz=imgsz, verbose=False, save=False,
        )
        boxes = results[0].boxes
        preds = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls  = boxes.cls.cpu().numpy().astype(int)
            preds = [(model_names[c], tuple(b)) for c, b in zip(cls, xyxy)]

        used_pred = set()
        for true_cls, gt_box in gt_boxes:
            best_iou, best_idx = 0.0, -1
            for i, (_, p_box) in enumerate(preds):
                if i in used_pred:
                    continue
                v = iou(gt_box, p_box)
                if v > best_iou:
                    best_iou, best_idx = v, i
            if best_iou >= iou_match and best_idx >= 0:
                pred_cls = preds[best_idx][0]
                used_pred.add(best_idx)
            else:
                pred_cls = "No detection"
            counts[true_cls][pred_cls] += 1

        for i, (pred_cls, _) in enumerate(preds):
            if i not in used_pred:
                false_positives[pred_cls] += 1

    if skipped:
        print(f"Skipped {skipped} images (could not parse subclass)\n")

    # ---- Console table ----
    col_w = max(len(c) for c in pred_labels) + 3
    header = f"{'True \\ Pred':<13}" + "".join(f"{c:>{col_w}}" for c in pred_labels)
    print(header)
    print("-" * len(header))
    for t in TRUE_CLASSES:
        row = f"{t:<13}" + "".join(f"{counts[t].get(c, 0):>{col_w}}" for c in pred_labels)
        print(row)

    print("\nFalse positives (predicted boxes with no GT match):")
    for c in model_names:
        print(f"  {c:<12} {false_positives.get(c, 0)}")

    # ---- Heat-map ----
    matrix = np.array([[counts[t].get(c, 0) for c in pred_labels] for t in TRUE_CLASSES],
                      dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_sums, where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, data, title, fmt in [
        (axes[0], matrix, "Counts",                   ".0f"),
        (axes[1], norm,   "Row-normalised (recall)",  ".2f"),
    ]:
        im = ax.imshow(data, cmap="Blues", aspect="auto",
                       vmin=0, vmax=data.max() if data.max() > 0 else 1)
        ax.set_xticks(range(len(pred_labels)))
        ax.set_xticklabels(pred_labels, rotation=35, ha="right")
        ax.set_yticks(range(len(TRUE_CLASSES)))
        ax.set_yticklabels(TRUE_CLASSES)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title)
        for i in range(len(TRUE_CLASSES)):
            for j in range(len(pred_labels)):
                v = data[i, j]
                color = "white" if v > data.max() * 0.6 else "black"
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        fontsize=9, color=color)
        fig.colorbar(im, ax=ax)

    fig.suptitle(f"Granular WBC model — box-level confusion (IoU≥{iou_match})",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = Path(__file__).parent / "granular_confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--source-dir", default=str(SOURCE_DIR))
    p.add_argument("--conf",       type=float, default=0.25)
    p.add_argument("--iou-match",  type=float, default=IOU_MATCH)
    p.add_argument("--imgsz",      type=int,   default=640)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(Path(a.checkpoint), Path(a.source_dir), a.conf, a.iou_match, a.imgsz)
