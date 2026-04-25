"""Run inference on every Raabin-WBC + wbc training image not yet processed,
keep only images with **exactly one** WBC detected, correct its class with
the ground-truth from the filename, and merge the result into the existing
wbc_merged.zip → produces wbc_final_merged.zip.

Rules per image (after filtering background RBC/Platelet boxes):
  - 0 WBC detected   → skip
  - >1 WBC detected  → skip
  - 1 WBC, correct   → keep as-is
  - 1 WBC, wrong     → relabel that single WBC box to the true subclass
"""

import io
import json
import zipfile
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RAABIN_DIR    = PROJECT_ROOT / "Raabin-WBC" / "Train"
RAABIN_JSON   = PROJECT_ROOT / "Raabin-WBC" / "Train.json"
WBC_DIR       = PROJECT_ROOT / "wbc" / "Train"
EXISTING_DIR  = PROJECT_ROOT / "wbc_detection_finetuning"
MERGED_ZIP    = PROJECT_ROOT / "cvat_review" / "wbc_merged.zip"
CHECKPOINT    = PROJECT_ROOT / "runs" / "finetune" / "granular_wbc_ft" / "weights" / "best.pt"
OUT_ZIP       = PROJECT_ROOT / "cvat_review" / "wbc_final_merged.zip"

WBC_SUBCLASSES   = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]
TARGET_CLASSES   = ["RBC", "Platelets", "Neutrophil", "Lymphocyte",
                    "Monocyte", "Eosinophil", "Basophil"]
RAABIN_CLASS_MAP = {1: "Neutrophil", 2: "Lymphocyte", 3: "Monocyte",
                    4: "Eosinophil", 5: "Basophil"}

CONF      = 0.25
IOU       = 0.7
IMG_SIZE  = 640
BATCH     = 32


# ---------------------------------------------------------------------------
def already_processed_basenames() -> set[str]:
    """Names from wbc_detection_finetuning/ AND from wbc_merged.zip.

    Filenames in those buckets carry a 'raabin_<sub>_' or 'wbc_<sub>_' prefix;
    strip it to recover the original Raabin/wbc basename.
    """
    used: set[str] = set()

    def strip_prefix(stem: str) -> str:
        parts = stem.split("_", 2)
        if len(parts) >= 3 and parts[0] in {"raabin", "wbc"} and parts[1] in WBC_SUBCLASSES:
            return parts[2]
        return stem

    if EXISTING_DIR.is_dir():
        for p in EXISTING_DIR.iterdir():
            used.add(strip_prefix(p.stem))
            used.add(strip_prefix(p.stem) + p.suffix)

    if MERGED_ZIP.exists():
        with zipfile.ZipFile(MERGED_ZIP) as zf:
            for member in zf.namelist():
                if member.startswith("obj_train_data/") and not member.endswith(".txt"):
                    name = Path(member).name
                    stem = Path(name).stem
                    base = strip_prefix(stem)
                    used.add(base)
                    used.add(base + ".jpg")
                    used.add(base + ".png")

    return used


def collect_candidates(excluded: set[str]):
    """yield (src_path, dest_stem, true_subclass)."""
    # Raabin
    if RAABIN_JSON.exists():
        labels = json.load(open(RAABIN_JSON))
        for fname, cid in labels.items():
            sub = RAABIN_CLASS_MAP.get(cid)
            if sub is None or fname in excluded:
                continue
            stem = Path(fname).stem
            yield RAABIN_DIR / fname, f"raabin_{sub}_{stem}", sub

    # wbc
    if WBC_DIR.is_dir():
        for sub in WBC_SUBCLASSES:
            cls_dir = WBC_DIR / sub
            if not cls_dir.is_dir():
                continue
            for img in cls_dir.glob("*.jpg"):
                if img.name in excluded:
                    continue
                yield img, f"wbc_{sub}_{img.stem}", sub


# ---------------------------------------------------------------------------
def yolo_label_from_result(result, model_names: list[str], true_sub: str) -> str | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    h, w = result.orig_shape
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    xyxy    = boxes.xyxy.cpu().numpy()

    wbc_indices = [i for i, c in enumerate(cls_ids)
                   if model_names[c] in WBC_SUBCLASSES]
    if len(wbc_indices) != 1:
        return None

    wbc_target_id = TARGET_CLASSES.index(true_sub)

    lines = []
    for i, c in enumerate(cls_ids):
        x1, y1, x2, y2 = xyxy[i]
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        cls_id = wbc_target_id if i in wbc_indices else c
        lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main():
    try:
        from ultralytics import YOLO
        from PIL import Image
    except ImportError:
        raise SystemExit("Run: pip install ultralytics pillow")

    excluded = already_processed_basenames()
    print(f"Excluding {len(excluded)} already-processed basenames")

    candidates = list(collect_candidates(excluded))
    print(f"Candidates to process: {len(candidates)}")

    print(f"\nLoading model: {CHECKPOINT}")
    model = YOLO(str(CHECKPOINT))
    model_names = [model.names[i] for i in sorted(model.names)]
    if model_names != TARGET_CLASSES:
        print(f"WARNING: model classes {model_names} differ from {TARGET_CLASSES}")

    # ---- Stream new entries directly into the output ZIP ----
    print(f"\nWriting merged output → {OUT_ZIP}")
    obj_data = (
        f"classes = {len(TARGET_CLASSES)}\n"
        f"names = obj.names\n"
        f"train = train.txt\n"
        f"backup = backup/\n"
    )
    image_paths: list[str] = []
    seen: set[str] = set()

    n_correct = n_relabeled = n_skipped_zero = n_skipped_multi = 0

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as out:
        out.writestr("obj.names", "\n".join(TARGET_CLASSES))
        out.writestr("obj.data", obj_data)

        # 1. Carry everything from wbc_merged.zip
        if MERGED_ZIP.exists():
            with zipfile.ZipFile(MERGED_ZIP) as src_zf:
                for m in src_zf.namelist():
                    if m in {"obj.names", "obj.data", "train.txt"} or m.endswith("/"):
                        continue
                    data = src_zf.read(m)
                    out.writestr(m, data)
                    if m.startswith("obj_train_data/") and not m.endswith(".txt"):
                        image_paths.append(m)
                        seen.add(Path(m).stem)
            print(f"  carried {len(image_paths)} images from wbc_merged.zip")

        # 2. Stream-process candidates in batches
        for batch_start in range(0, len(candidates), BATCH):
            batch = candidates[batch_start:batch_start + BATCH]
            sources = [str(c[0]) for c in batch]
            results = model.predict(
                source=sources, conf=CONF, iou=IOU, imgsz=IMG_SIZE,
                verbose=False, save=False, device=0, stream=False,
            )

            for (src, dest_stem, true_sub), result in zip(batch, results):
                if dest_stem in seen:
                    continue
                boxes = getattr(result, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    n_skipped_zero += 1
                    continue
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                wbc_count = sum(1 for c in cls_ids if model_names[c] in WBC_SUBCLASSES)
                if wbc_count == 0:
                    n_skipped_zero += 1
                    continue
                if wbc_count > 1:
                    n_skipped_multi += 1
                    continue

                # Determine if predicted WBC subclass matches the true one
                pred_wbc_idx = next(i for i, c in enumerate(cls_ids)
                                    if model_names[c] in WBC_SUBCLASSES)
                pred_wbc_sub = model_names[cls_ids[pred_wbc_idx]]
                if pred_wbc_sub == true_sub:
                    n_correct += 1
                else:
                    n_relabeled += 1

                label_text = yolo_label_from_result(result, model_names, true_sub)
                if label_text is None:
                    continue

                # Convert source image to PNG and write
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    out.writestr(f"obj_train_data/{dest_stem}.png", buf.getvalue())
                out.writestr(f"obj_train_data/{dest_stem}.txt", label_text)
                image_paths.append(f"obj_train_data/{dest_stem}.png")
                seen.add(dest_stem)

            done = min(batch_start + BATCH, len(candidates))
            print(f"  {done}/{len(candidates)}  "
                  f"kept-correct={n_correct}  relabeled={n_relabeled}  "
                  f"skip-zero={n_skipped_zero}  skip-multi={n_skipped_multi}  "
                  f"total-in-zip={len(image_paths)}",
                  flush=True)

        out.writestr("train.txt", "\n".join(image_paths))

    print()
    print(f"Kept correct (1 WBC, matches)    : {n_correct}")
    print(f"Relabeled    (1 WBC, mismatch)   : {n_relabeled}")
    print(f"Skipped 0 WBC                    : {n_skipped_zero}")
    print(f"Skipped >1 WBC                   : {n_skipped_multi}")
    print(f"Total images in merged ZIP        : {len(image_paths)}")
    print(f"\nDone → {OUT_ZIP}")


if __name__ == "__main__":
    main()
