"""Fine-tune the TXL-PBC YOLO model on the wbc_reviewed dataset.

Dataset source: wbc_reviewed/obj_train_data/obj_train_data/
  - Images and YOLO label .txt files live side by side.
  - obj.names class order: RBC=0, WBC=1, Platelets=2
  - Original model class order: WBC=0, RBC=1, Platelets=2
  - Labels are remapped so the fine-tuned model stays consistent
    with the original class IDs.

Output:
  finetune_dataset/
    images/train/   images/val/
    labels/train/   labels/val/
    data.yaml
"""

import argparse
import random
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR   = PROJECT_ROOT / "wbc_reviewed" / "obj_train_data" / "obj_train_data"
DATASET_DIR  = PROJECT_ROOT / "finetune_dataset"
CHECKPOINT   = PROJECT_ROOT / "runs" / "yolo26" / "txl_pbc_yolo26m2" / "weights" / "last.pt"

# Class order in the reviewed labels (from CVAT export)
CVAT_CLASSES   = ["RBC", "WBC", "Platelets"]
# Class order the original model uses — fine-tuned model will keep this order
TARGET_CLASSES = ["WBC", "RBC", "Platelets"]

# Remap: cvat_id -> target_id
CLASS_REMAP = {
    CVAT_CLASSES.index(c): TARGET_CLASSES.index(c) for c in TARGET_CLASSES
}

VAL_SPLIT  = 0.20
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Training hyper-parameters (override via CLI)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    epochs   = 50,
    imgsz    = 640,
    batch    = 16,
    lr0      = 1e-4,
    lrf      = 0.01,
    patience = 15,
    device   = "",          # "" = auto
    project  = str(PROJECT_ROOT / "runs" / "finetune"),
    name     = "wbc_reviewed_ft",
    workers  = 4,
    cos_lr   = True,
    freeze   = 10,          # freeze first N backbone layers
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def remap_label_file(src: Path, dst: Path) -> None:
    lines = src.read_text().strip().splitlines()
    remapped = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        parts[0] = str(CLASS_REMAP[int(parts[0])])
        remapped.append(" ".join(parts))
    dst.write_text("\n".join(remapped))


def build_dataset(val_split: float, seed: int) -> Path:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    pairs = []
    for img_path in sorted(SOURCE_DIR.iterdir()):
        if img_path.suffix.lower() not in image_exts:
            continue
        lbl_path = img_path.with_suffix(".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))

    if not pairs:
        raise FileNotFoundError(f"No image/label pairs found in {SOURCE_DIR}")

    random.seed(seed)
    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * val_split))
    val_set = set(range(len(pairs) - n_val, len(pairs)))

    print(f"Found {len(pairs)} pairs → train: {len(pairs)-n_val}  val: {n_val}")

    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    for idx, (img, lbl) in enumerate(pairs):
        split = "val" if idx in val_set else "train"
        shutil.copy2(img, DATASET_DIR / "images" / split / img.name)
        remap_label_file(lbl, DATASET_DIR / "labels" / split / lbl.name)

    yaml_path = DATASET_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {DATASET_DIR}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\nnc: {len(TARGET_CLASSES)}\n"
        f"names: {TARGET_CLASSES}\n"
    )
    print(f"Dataset written to {DATASET_DIR}")
    print(f"data.yaml: {yaml_path}")
    return yaml_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune TXL-PBC YOLO on wbc_reviewed")
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--epochs",     type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--imgsz",      type=int,   default=DEFAULTS["imgsz"])
    p.add_argument("--batch",      type=int,   default=DEFAULTS["batch"])
    p.add_argument("--lr0",        type=float, default=DEFAULTS["lr0"])
    p.add_argument("--lrf",        type=float, default=DEFAULTS["lrf"])
    p.add_argument("--patience",   type=int,   default=DEFAULTS["patience"])
    p.add_argument("--device",     default=DEFAULTS["device"])
    p.add_argument("--project",    default=DEFAULTS["project"])
    p.add_argument("--name",       default=DEFAULTS["name"])
    p.add_argument("--workers",    type=int,   default=DEFAULTS["workers"])
    p.add_argument("--freeze",     type=int,   default=DEFAULTS["freeze"],
                   help="Freeze first N backbone layers (0 = no freeze)")
    p.add_argument("--no-cos-lr",  dest="cos_lr", action="store_false",
                   help="Disable cosine LR schedule")
    p.add_argument("--rebuild",    action="store_true",
                   help="Rebuild dataset even if finetune_dataset/ already exists")
    p.add_argument("--val-split",  type=float, default=VAL_SPLIT)
    p.add_argument("--seed",       type=int,   default=RANDOM_SEED)
    p.set_defaults(cos_lr=DEFAULTS["cos_lr"])
    return p.parse_args()


def main():
    args = parse_args()

    # Build / reuse dataset
    yaml_path = DATASET_DIR / "data.yaml"
    if args.rebuild or not yaml_path.exists():
        if DATASET_DIR.exists():
            shutil.rmtree(DATASET_DIR)
        yaml_path = build_dataset(args.val_split, args.seed)
    else:
        print(f"Reusing existing dataset at {DATASET_DIR} (pass --rebuild to recreate)")

    # Load model
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not found. Run: pip install ultralytics")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"\nFine-tuning from: {checkpoint}")
    model = YOLO(str(checkpoint))

    train_kwargs = dict(
        data      = str(yaml_path),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        lr0       = args.lr0,
        lrf       = args.lrf,
        patience  = args.patience,
        project   = args.project,
        name      = args.name,
        workers   = args.workers,
        cos_lr    = args.cos_lr,
        freeze    = args.freeze if args.freeze > 0 else None,
        exist_ok  = True,
        verbose   = True,
    )
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    print("\nTraining complete.")
    print(f"Best checkpoint: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    return results


if __name__ == "__main__":
    main()
