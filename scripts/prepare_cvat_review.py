"""Sample new WBC images, run inference with the granular model, and split
into two CVAT-ready datasets for manual review:

  - "problematic"  : no WBC detected, OR a WBC of a different subclass detected
  - "correct"      : at least one WBC of the expected subclass detected

Output (zipped, YOLO 1.1 / CVAT-compatible):
  cvat_review_problematic.zip
  cvat_review_correct.zip
"""

import io
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAABIN_DIR   = PROJECT_ROOT / "Raabin-WBC" / "Train"
RAABIN_JSON  = PROJECT_ROOT / "Raabin-WBC" / "Train.json"
WBC_DIR      = PROJECT_ROOT / "wbc" / "Train"
EXISTING_DIR = PROJECT_ROOT / "wbc_detection_finetuning"
CHECKPOINT   = PROJECT_ROOT / "runs" / "finetune" / "granular_wbc_ft" / "weights" / "best.pt"

PER_CLASS_PER_DATASET = 40              # 40 × 5 classes × 2 datasets ≈ 400 new images
RANDOM_SEED           = 7
CONF_THRESHOLD        = 0.25
IMG_SIZE              = 640

WBC_SUBCLASSES = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]
RAABIN_CLASS_MAP = {1: "Neutrophil", 2: "Lymphocyte", 3: "Monocyte",
                    4: "Eosinophil", 5: "Basophil"}

# --- target class order (matches the granular model) ---
TARGET_CLASSES = ["RBC", "Platelets", "Neutrophil", "Lymphocyte",
                  "Monocyte", "Eosinophil", "Basophil"]

OUT_DIR = PROJECT_ROOT / "cvat_review"


# ---------------------------------------------------------------------------
def already_sampled_basenames() -> set[str]:
    """Return the original-filename set for images already in wbc_detection_finetuning/.

    Names are stored as e.g. 'raabin_Basophil_<original.jpg>' or
    'wbc_Lymphocyte_<original.jpg>'. We strip the first two prefix tokens.
    """
    used = set()
    for p in EXISTING_DIR.iterdir():
        parts = p.name.split("_", 2)
        if len(parts) >= 3:
            used.add(parts[2])
    return used


def sample_raabin(per_class: int, seed: int, exclude: set[str]):
    labels = json.load(open(RAABIN_JSON))   # {filename: class_id}
    by_class: dict[str, list[str]] = defaultdict(list)
    for fname, cid in labels.items():
        sub = RAABIN_CLASS_MAP.get(cid)
        if sub and fname not in exclude:
            by_class[sub].append(fname)

    rng = random.Random(seed)
    out = []
    for sub in WBC_SUBCLASSES:
        pool = by_class[sub]
        n = min(per_class, len(pool))
        for fname in rng.sample(pool, n):
            out.append((RAABIN_DIR / fname, f"raabin_{sub}_{fname}", sub))
    return out


def sample_wbc(per_class: int, seed: int, exclude: set[str]):
    rng = random.Random(seed + 1)
    out = []
    for sub in WBC_SUBCLASSES:
        cls_dir = WBC_DIR / sub
        pool = [p for p in cls_dir.glob("*.jpg") if p.name not in exclude]
        n = min(per_class, len(pool))
        for img in rng.sample(pool, n):
            out.append((img, f"wbc_{sub}_{img.name}", sub))
    return out


# ---------------------------------------------------------------------------
def run_inference(model, image_path: Path, conf: float, imgsz: int):
    results = model.predict(
        source=str(image_path), conf=conf, iou=0.7,
        imgsz=imgsz, verbose=False, save=False, device=0,
    )[0]
    boxes = getattr(results, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return [], (results.orig_shape[1], results.orig_shape[0])

    out = []
    for cls_id, conf_v, xyxy in zip(
        boxes.cls.cpu().tolist(),
        boxes.conf.cpu().tolist(),
        boxes.xyxy.cpu().tolist(),
    ):
        out.append((int(cls_id), float(conf_v), tuple(xyxy)))
    h, w = results.orig_shape
    return out, (w, h)


def detections_to_yolo_lines(dets, img_w: int, img_h: int) -> str:
    lines = []
    for cls_id, _, (x1, y1, x2, y2) in dets:
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def build_cvat_zip(entries, out_path: Path):
    """entries: list of (src_image_path, dest_name_stem, label_text)"""
    image_paths = [f"obj_train_data/{stem}.png" for _, stem, _ in entries]
    obj_data = (
        f"classes = {len(TARGET_CLASSES)}\n"
        f"names = obj.names\n"
        f"train = train.txt\n"
        f"backup = backup/\n"
    )

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("obj.names", "\n".join(TARGET_CLASSES))
        zf.writestr("obj.data", obj_data)
        zf.writestr("train.txt", "\n".join(image_paths))
        for src, stem, label in entries:
            from PIL import Image
            with Image.open(src) as im:
                im = im.convert("RGB")
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                zf.writestr(f"obj_train_data/{stem}.png", buf.getvalue())
            zf.writestr(f"obj_train_data/{stem}.txt", label)


# ---------------------------------------------------------------------------
def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Run: pip install ultralytics")

    excluded = already_sampled_basenames()
    print(f"Excluding {len(excluded)} already-sampled basenames")

    raabin = sample_raabin(PER_CLASS_PER_DATASET, RANDOM_SEED, excluded)
    wbc    = sample_wbc(PER_CLASS_PER_DATASET, RANDOM_SEED, excluded)
    samples = raabin + wbc
    print(f"Total new samples: {len(samples)}  "
          f"(raabin={len(raabin)}  wbc={len(wbc)})")

    print(f"\nLoading model: {CHECKPOINT}")
    model = YOLO(str(CHECKPOINT))
    model_names = [model.names[i] for i in sorted(model.names)]
    if model_names != TARGET_CLASSES:
        print(f"WARNING: model classes {model_names} differ from TARGET_CLASSES {TARGET_CLASSES}")

    OUT_DIR.mkdir(exist_ok=True)
    problematic, correct = [], []

    for i, (src, dest_name, true_sub) in enumerate(samples, 1):
        dets, (img_w, img_h) = run_inference(model, src, CONF_THRESHOLD, IMG_SIZE)
        detected_subs = {model_names[c] for c, _, _ in dets if model_names[c] in WBC_SUBCLASSES}

        if true_sub in detected_subs:
            bucket = correct
        else:
            bucket = problematic   # no WBC, OR wrong WBC subclass

        stem = Path(dest_name).stem
        bucket.append((src, stem, detections_to_yolo_lines(dets, img_w, img_h)))

        if i % 25 == 0 or i == len(samples):
            print(f"  {i}/{len(samples)}  "
                  f"correct={len(correct)}  problematic={len(problematic)}")

    # ---- Per-subclass breakdown ----
    counts: dict[str, dict[str, int]] = {s: {"correct": 0, "problematic": 0} for s in WBC_SUBCLASSES}
    for src, stem, _ in correct:
        counts[stem.split("_")[1]]["correct"] += 1
    for src, stem, _ in problematic:
        counts[stem.split("_")[1]]["problematic"] += 1

    print("\nPer-subclass split:")
    print(f"  {'Subclass':<12}  correct  problematic")
    for s in WBC_SUBCLASSES:
        print(f"  {s:<12}  {counts[s]['correct']:>7}  {counts[s]['problematic']:>11}")

    # ---- Write ZIPs ----
    out_problematic = OUT_DIR / "cvat_review_problematic.zip"
    out_correct     = OUT_DIR / "cvat_review_correct.zip"

    print(f"\nBuilding {out_problematic.name} ({len(problematic)} images)…")
    build_cvat_zip(problematic, out_problematic)
    print(f"Building {out_correct.name} ({len(correct)} images)…")
    build_cvat_zip(correct, out_correct)

    print(f"\nDone. Open ZIPs in CVAT:")
    print(f"  Problematic : {out_problematic}")
    print(f"  Correct     : {out_correct}")


if __name__ == "__main__":
    main()
