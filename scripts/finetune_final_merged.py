"""Second-round fine-tune on cvat_review/wbc_final_merged.zip (9k images).

  - Starts from runs/finetune/granular_wbc_ft/weights/best.pt
  - Same 7-class label set: RBC, Platelets + 5 WBC subclasses
  - 80/20 train/val split
  - On-the-fly augmentation (mosaic, mixup, flips, HSV, copy-paste)

The source ZIP follows YOLO 1.1 / CVAT layout:
    obj.names
    obj.data
    train.txt
    obj_train_data/
        image1.png
        image1.txt
        ...

We extract once to finetune_v2_dataset_src/ (skipped on subsequent runs unless
--rebuild is passed), then build train/val splits in finetune_v2_dataset/.
"""

import argparse
import random
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ZIP   = PROJECT_ROOT / "cvat_review" / "wbc_final_merged.zip"
EXTRACT_DIR  = PROJECT_ROOT / "finetune_v2_dataset_src"
DATASET_DIR  = PROJECT_ROOT / "finetune_v2_dataset"
CHECKPOINT   = PROJECT_ROOT / "runs" / "finetune" / "granular_wbc_ft" / "weights" / "best.pt"

TARGET_CLASSES = ["RBC", "Platelets", "Neutrophil", "Lymphocyte",
                  "Monocyte", "Eosinophil", "Basophil"]

VAL_SPLIT   = 0.20
RANDOM_SEED = 42

DEFAULTS = dict(
    epochs   = 60,
    imgsz    = 640,
    batch    = 16,
    lr0      = 3e-4,
    lrf      = 0.01,
    patience = 15,
    device   = "0",
    project  = str(PROJECT_ROOT / "runs" / "finetune"),
    name     = "granular_wbc_ft_v2",
    workers  = 4,
    freeze   = 10,
    # augmentation
    mosaic     = 1.0,
    mixup      = 0.15,
    flipud     = 0.5,
    fliplr     = 0.5,
    degrees    = 15.0,
    scale      = 0.5,
    hsv_h      = 0.015,
    hsv_s      = 0.7,
    hsv_v      = 0.4,
    copy_paste = 0.1,
)


# ---------------------------------------------------------------------------
def extract_source(rebuild: bool = False) -> Path:
    """Extract obj_train_data/ from SOURCE_ZIP into EXTRACT_DIR."""
    obj_dir = EXTRACT_DIR / "obj_train_data"
    if obj_dir.exists() and not rebuild:
        n = sum(1 for _ in obj_dir.glob("*.png")) + sum(1 for _ in obj_dir.glob("*.jpg"))
        print(f"Reusing extracted source ({n} images at {obj_dir})  --rebuild to redo")
        return obj_dir

    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True)

    print(f"Extracting {SOURCE_ZIP.name} → {EXTRACT_DIR}")
    with zipfile.ZipFile(SOURCE_ZIP) as zf:
        members = [m for m in zf.namelist() if m.startswith("obj_train_data/")]
        zf.extractall(EXTRACT_DIR, members=members)
    print(f"Extracted {len(members)} files")
    return obj_dir


def build_dataset(src_dir: Path, val_split: float, seed: int) -> Path:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pairs = []
    for img in sorted(src_dir.iterdir()):
        if img.suffix.lower() not in image_exts:
            continue
        lbl = img.with_suffix(".txt")
        if lbl.exists():
            pairs.append((img, lbl))

    if not pairs:
        raise FileNotFoundError(f"No image/label pairs in {src_dir}")

    random.seed(seed)
    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * val_split))
    val_idx = set(range(len(pairs) - n_val, len(pairs)))

    print(f"Pairs: {len(pairs)}  →  train: {len(pairs)-n_val}  val: {n_val}")

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    for idx, (img, lbl) in enumerate(pairs):
        split = "val" if idx in val_idx else "train"
        shutil.copy2(img, DATASET_DIR / "images" / split / img.name)
        shutil.copy2(lbl, DATASET_DIR / "labels" / split / lbl.name)

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
    p.add_argument("--rebuild",  action="store_true",
                   help="Re-extract source ZIP and rebuild dataset")
    p.add_argument("--val-split",type=float, default=VAL_SPLIT)
    p.add_argument("--seed",     type=int,   default=RANDOM_SEED)
    return p.parse_args()


def main():
    args = parse_args()

    if not SOURCE_ZIP.exists():
        raise SystemExit(f"Source ZIP missing: {SOURCE_ZIP}")

    src_dir   = extract_source(rebuild=args.rebuild)
    yaml_path = DATASET_DIR / "data.yaml"
    if args.rebuild or not yaml_path.exists():
        yaml_path = build_dataset(src_dir, args.val_split, args.seed)
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
