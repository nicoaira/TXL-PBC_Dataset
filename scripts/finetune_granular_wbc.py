"""Fine-tune YOLO with granular WBC subclasses.

New class set (7 total):
  0: RBC          1: Platelets
  2: Neutrophil   3: Lymphocyte   4: Monocyte
  5: Eosinophil   6: Basophil

Source: wbc_reviewed/obj_train_data/obj_train_data/
  CVAT class IDs (from obj.names): 0=RBC  1=WBC  2=Platelets

Label remapping strategy
-------------------------
For each image the WBC subclass is read from the filename
(pattern: {source}_{Subclass}_{rest}.ext).

  - Largest bounding box  →  specific WBC subclass (ground-truth from filename)
  - Other boxes with CVAT class 1 (WBC)  →  same specific WBC subclass
  - Other boxes with CVAT class 0 (RBC)  →  target class 0  (RBC)
  - Other boxes with CVAT class 2 (Platelets)  →  target class 1  (Platelets)
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR   = PROJECT_ROOT / "wbc_reviewed" / "obj_train_data" / "obj_train_data"
DATASET_DIR  = PROJECT_ROOT / "finetune_granular_dataset"
CHECKPOINT   = PROJECT_ROOT / "runs" / "yolo26" / "txl_pbc_yolo26m2" / "weights" / "last.pt"

WBC_SUBCLASSES = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]
TARGET_CLASSES = ["RBC", "Platelets", "Neutrophil", "Lymphocyte",
                  "Monocyte", "Eosinophil", "Basophil"]

# CVAT class id → target class id for non-WBC boxes
CVAT_FIXED_REMAP = {0: 0, 2: 1}   # RBC→0, Platelets→1

VAL_SPLIT   = 0.20
RANDOM_SEED = 42

DEFAULTS = dict(
    epochs   = 80,
    imgsz    = 640,
    batch    = 16,
    lr0      = 5e-4,
    lrf      = 0.01,
    patience = 20,
    device   = "0",
    project  = str(PROJECT_ROOT / "runs" / "finetune"),
    name     = "granular_wbc_ft",
    workers  = 4,
    freeze   = 10,
    # augmentation
    mosaic   = 1.0,
    mixup    = 0.15,
    flipud   = 0.5,
    fliplr   = 0.5,
    degrees  = 15.0,
    scale    = 0.5,
    hsv_h    = 0.015,
    hsv_s    = 0.7,
    hsv_v    = 0.4,
    copy_paste = 0.1,
)

# ---------------------------------------------------------------------------

def wbc_subclass_from_stem(stem: str):
    parts = stem.split("_")
    if len(parts) >= 2 and parts[1] in WBC_SUBCLASSES:
        return parts[1]
    return None


def remap_labels(src_txt: Path, dst_txt: Path, wbc_subclass: str) -> None:
    wbc_target_id = TARGET_CLASSES.index(wbc_subclass)
    lines = [l for l in src_txt.read_text().splitlines() if l.strip()]
    if not lines:
        dst_txt.write_text("")
        return

    # Parse boxes, find largest by area
    parsed = []
    for line in lines:
        parts = line.split()
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        parsed.append((cls_id, cx, cy, w, h))

    areas  = [p[3] * p[4] for p in parsed]
    largest = int(np.argmax(areas))

    remapped = []
    for i, (cls_id, cx, cy, w, h) in enumerate(parsed):
        if i == largest:
            new_cls = wbc_target_id
        elif cls_id == 1:          # CVAT WBC → also specific subclass
            new_cls = wbc_target_id
        else:
            new_cls = CVAT_FIXED_REMAP.get(cls_id)
            if new_cls is None:
                continue           # unknown class; skip
        remapped.append(f"{new_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    dst_txt.write_text("\n".join(remapped))


def build_dataset(val_split: float, seed: int) -> Path:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    pairs = []
    for img in sorted(SOURCE_DIR.iterdir()):
        if img.suffix.lower() not in image_exts:
            continue
        lbl = img.with_suffix(".txt")
        subclass = wbc_subclass_from_stem(img.stem)
        if lbl.exists() and subclass:
            pairs.append((img, lbl, subclass))

    if not pairs:
        raise FileNotFoundError(f"No valid image/label pairs in {SOURCE_DIR}")

    random.seed(seed)
    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * val_split))
    val_idx = set(range(len(pairs) - n_val, len(pairs)))

    print(f"Pairs: {len(pairs)}  →  train: {len(pairs)-n_val}  val: {n_val}")

    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    for idx, (img, lbl, subclass) in enumerate(pairs):
        split = "val" if idx in val_idx else "train"
        shutil.copy2(img, DATASET_DIR / "images" / split / img.name)
        remap_labels(lbl, DATASET_DIR / "labels" / split / lbl.name, subclass)

    yaml_path = DATASET_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {DATASET_DIR}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\nnc: {len(TARGET_CLASSES)}\n"
        f"names: {TARGET_CLASSES}\n"
    )
    print(f"Dataset → {DATASET_DIR}")
    return yaml_path


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--epochs",   type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--imgsz",    type=int,   default=DEFAULTS["imgsz"])
    p.add_argument("--batch",    type=int,   default=DEFAULTS["batch"])
    p.add_argument("--lr0",      type=float, default=DEFAULTS["lr0"])
    p.add_argument("--patience", type=int,   default=DEFAULTS["patience"])
    p.add_argument("--device",   default=DEFAULTS["device"])
    p.add_argument("--project",  default=DEFAULTS["project"])
    p.add_argument("--name",     default=DEFAULTS["name"])
    p.add_argument("--freeze",   type=int,   default=DEFAULTS["freeze"])
    p.add_argument("--rebuild",  action="store_true")
    p.add_argument("--val-split",type=float, default=VAL_SPLIT)
    p.add_argument("--seed",     type=int,   default=RANDOM_SEED)
    return p.parse_args()


def main():
    args = parse_args()

    yaml_path = DATASET_DIR / "data.yaml"
    if args.rebuild or not yaml_path.exists():
        if DATASET_DIR.exists():
            shutil.rmtree(DATASET_DIR)
        yaml_path = build_dataset(args.val_split, args.seed)
    else:
        print(f"Reusing dataset at {DATASET_DIR}  (--rebuild to recreate)")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Run: pip install ultralytics")

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    print(f"\nCheckpoint : {ckpt}")
    print(f"Classes    : {TARGET_CLASSES}\n")
    model = YOLO(str(ckpt))

    model.train(
        data     = str(yaml_path),
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        lr0      = args.lr0,
        lrf      = DEFAULTS["lrf"],
        patience = args.patience,
        project  = args.project,
        name     = args.name,
        workers  = DEFAULTS["workers"],
        freeze   = args.freeze if args.freeze > 0 else None,
        device   = args.device,
        cos_lr   = True,
        exist_ok = True,
        verbose  = True,
        # --- augmentation ---
        mosaic     = DEFAULTS["mosaic"],
        mixup      = DEFAULTS["mixup"],
        flipud     = DEFAULTS["flipud"],
        fliplr     = DEFAULTS["fliplr"],
        degrees    = DEFAULTS["degrees"],
        scale      = DEFAULTS["scale"],
        hsv_h      = DEFAULTS["hsv_h"],
        hsv_s      = DEFAULTS["hsv_s"],
        hsv_v      = DEFAULTS["hsv_v"],
        copy_paste = DEFAULTS["copy_paste"],
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nDone. Best checkpoint → {best}")


if __name__ == "__main__":
    main()
