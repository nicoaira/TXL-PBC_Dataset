"""Confusion matrix: true WBC subclass vs model-predicted class.

For every image in wbc_detection_finetuning/ the true WBC subclass is read
from the filename (pattern: {source}_{WBCClass}_{rest}.ext).
The model predicts bounding boxes; the dominant class (highest-confidence
detection) is taken as the image-level prediction.  If the model fires no
box at all the image is counted as 'No detection'.

Output
------
  - Console: per-class counts and confusion matrix
  - wbc_confusion_matrix.png  saved next to this script
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR    = PROJECT_ROOT / "wbc_detection_finetuning"
CHECKPOINT   = PROJECT_ROOT / "runs" / "finetune" / "granular_wbc_ft" / "weights" / "best.pt"

WBC_SUBCLASSES = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]


def true_class_from_filename(name: str) -> str | None:
    """Extract WBC subclass from filename like raabin_Neutrophil_... or wbc_Eosinophil_..."""
    parts = name.split("_")
    if len(parts) < 2:
        return None
    candidate = parts[1]
    return candidate if candidate in WBC_SUBCLASSES else None


def dominant_prediction(result, model_names: list[str]) -> str:
    """Return the class label of the largest bounding box (by area).

    These are cropped single-cell images where the main WBC fills most of the
    frame. Largest box = the main cell; highest-confidence would pick background
    RBCs instead.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return "No detection"
    xyxy = boxes.xyxy.cpu()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    best_idx = int(areas.argmax())
    cls_id   = int(boxes.cls[best_idx])
    return model_names[cls_id] if cls_id < len(model_names) else str(cls_id)


def run(checkpoint: Path, image_dir: Path, conf: float, iou: float, imgsz: int):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not found. Run: pip install ultralytics")

    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        raise SystemExit("matplotlib not found. Run: pip install matplotlib")

    print(f"Loading model: {checkpoint}")
    model = YOLO(str(checkpoint))
    model_names = [model.names[i] for i in sorted(model.names)]
    pred_labels = model_names + ["No detection"]
    print(f"Model classes: {model_names}\n")

    images = sorted(p for p in image_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    print(f"Found {len(images)} images in {image_dir.name}/\n")

    # counts[true_subclass][predicted_model_class] = n
    counts: dict[str, dict[str, int]] = {
        sub: defaultdict(int) for sub in WBC_SUBCLASSES
    }
    skipped = 0

    for img_path in images:
        true_cls = true_class_from_filename(img_path.stem)
        if true_cls is None:
            skipped += 1
            continue

        results = model.predict(
            source=str(img_path),
            conf=conf, iou=iou, imgsz=imgsz,
            verbose=False, save=False,
        )
        pred_cls = dominant_prediction(results[0], model_names)
        counts[true_cls][pred_cls] += 1

    if skipped:
        print(f"Skipped {skipped} images (could not parse class from filename)\n")

    # ---- Console summary ----
    col_w = max(len(c) for c in pred_labels) + 4
    header = f"{'True \\ Pred':<14}" + "".join(f"{c:>{col_w}}" for c in pred_labels)
    print(header)
    print("-" * len(header))
    for sub in WBC_SUBCLASSES:
        row = f"{sub:<14}" + "".join(
            f"{counts[sub].get(c, 0):>{col_w}}" for c in pred_labels
        )
        print(row)

    # ---- Build matrix for plotting ----
    matrix = np.array([
        [counts[sub].get(c, 0) for c in pred_labels]
        for sub in WBC_SUBCLASSES
    ], dtype=float)

    # Row-normalise (recall per true class)
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(matrix, row_sums, where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in [
        (axes[0], matrix,      "Counts",              ".0f"),
        (axes[1], norm_matrix, "Row-normalised (recall)", ".2f"),
    ]:
        im = ax.imshow(data, cmap="Blues", aspect="auto",
                       vmin=0, vmax=data.max() if data.max() > 0 else 1)
        ax.set_xticks(range(len(pred_labels)))
        ax.set_xticklabels(pred_labels, rotation=30, ha="right")
        ax.set_yticks(range(len(WBC_SUBCLASSES)))
        ax.set_yticklabels(WBC_SUBCLASSES)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True WBC subclass")
        ax.set_title(title)
        for i in range(len(WBC_SUBCLASSES)):
            for j in range(len(pred_labels)):
                val = data[i, j]
                txt = format(val, fmt)
                color = "white" if val > data.max() * 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        color=color, fontsize=10)
        fig.colorbar(im, ax=ax)

    fig.suptitle("WBC subclass confusion — model predictions on wbc_detection_finetuning",
                 fontsize=12, y=1.01)
    fig.tight_layout()

    out_path = Path(__file__).parent / "wbc_confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--image-dir",  default=str(IMAGE_DIR))
    p.add_argument("--conf",  type=float, default=0.25)
    p.add_argument("--iou",   type=float, default=0.70)
    p.add_argument("--imgsz", type=int,   default=640)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        checkpoint = Path(args.checkpoint),
        image_dir  = Path(args.image_dir),
        conf       = args.conf,
        iou        = args.iou,
        imgsz      = args.imgsz,
    )
