"""Merge two reviewed CVAT exports into a single YOLO 1.1 dataset:

  Input (in cvat_review/):
    - wbc_problematic_reviewed.zip   (CVAT for Images 1.1 / XML)
    - cvat_review_correct.zip        (YOLO 1.1)

  Output:
    - cvat_review/wbc_merged.zip     (YOLO 1.1 ready for CVAT/YOLO training)
"""

import io
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CVAT_DIR     = PROJECT_ROOT / "cvat_review"
PROBLEMATIC  = CVAT_DIR / "wbc_problematic_reviewed.zip"
CORRECT      = CVAT_DIR / "cvat_review_correct.zip"
OUT_ZIP      = CVAT_DIR / "wbc_merged.zip"

TARGET_CLASSES = ["RBC", "Platelets", "Neutrophil", "Lymphocyte",
                  "Monocyte", "Eosinophil", "Basophil"]


def normalise_image_name(name: str) -> str:
    """Strip 'images/' or 'obj_train_data/' prefix → just the basename."""
    return Path(name).name


# ---------------------------------------------------------------------------
def parse_problematic(zip_path: Path):
    """Yield (image_name, image_bytes, yolo_label_text) from CVAT XML export."""
    name_to_id = {n: i for i, n in enumerate(TARGET_CLASSES)}

    with zipfile.ZipFile(zip_path) as zf:
        xml_data = zf.read("annotations.xml")
        root = ET.fromstring(xml_data)

        # Map normalised image name → source path inside the zip
        zip_image_paths: dict[str, str] = {}
        for member in zf.namelist():
            if member.endswith("/") or member == "annotations.xml":
                continue
            zip_image_paths[normalise_image_name(member)] = member

        for img in root.findall("image"):
            name_in_xml = normalise_image_name(img.get("name"))
            w = float(img.get("width"))
            h = float(img.get("height"))

            yolo_lines = []
            for box in img.findall("box"):
                label = box.get("label")
                if label not in name_to_id:
                    continue
                cls_id = name_to_id[label]
                xtl = float(box.get("xtl"))
                ytl = float(box.get("ytl"))
                xbr = float(box.get("xbr"))
                ybr = float(box.get("ybr"))
                # clamp to image
                xtl = max(0.0, min(xtl, w)); xbr = max(0.0, min(xbr, w))
                ytl = max(0.0, min(ytl, h)); ybr = max(0.0, min(ybr, h))
                if xbr <= xtl or ybr <= ytl:
                    continue
                cx = ((xtl + xbr) / 2) / w
                cy = ((ytl + ybr) / 2) / h
                bw = (xbr - xtl) / w
                bh = (ybr - ytl) / h
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            zip_path_for_image = zip_image_paths.get(name_in_xml)
            if zip_path_for_image is None:
                print(f"  ! image referenced in XML but missing from zip: {name_in_xml}")
                continue

            img_bytes = zf.read(zip_path_for_image)
            yield name_in_xml, img_bytes, "\n".join(yolo_lines)


def parse_correct(zip_path: Path):
    """Yield (image_name, image_bytes, yolo_label_text) from YOLO 1.1 zip."""
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        # collect images under obj_train_data/
        for member in members:
            if not member.startswith("obj_train_data/"):
                continue
            if member.endswith("/"):
                continue
            stem = Path(member).stem
            ext  = Path(member).suffix.lower()
            if ext not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            label_member = f"obj_train_data/{stem}.txt"
            label_text = zf.read(label_member).decode() if label_member in members else ""
            yield Path(member).name, zf.read(member), label_text.strip()


# ---------------------------------------------------------------------------
def main():
    if not PROBLEMATIC.exists():
        raise SystemExit(f"missing: {PROBLEMATIC}")
    if not CORRECT.exists():
        raise SystemExit(f"missing: {CORRECT}")

    seen: set[str] = set()
    obj_data = (
        f"classes = {len(TARGET_CLASSES)}\n"
        f"names = obj.names\n"
        f"train = train.txt\n"
        f"backup = backup/\n"
    )

    image_paths: list[str] = []
    n_problematic = n_correct = n_dup = 0

    print(f"Building merged ZIP at {OUT_ZIP}")
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as out:
        out.writestr("obj.names", "\n".join(TARGET_CLASSES))
        out.writestr("obj.data", obj_data)

        # Problematic first (manually reviewed; trumps duplicates)
        print(f"  reading {PROBLEMATIC.name} (XML)…")
        for name, img_bytes, label_text in parse_problematic(PROBLEMATIC):
            stem = Path(name).stem
            if stem in seen:
                n_dup += 1
                continue
            seen.add(stem)
            out.writestr(f"obj_train_data/{stem}.png", img_bytes)
            out.writestr(f"obj_train_data/{stem}.txt", label_text)
            image_paths.append(f"obj_train_data/{stem}.png")
            n_problematic += 1

        print(f"  reading {CORRECT.name} (YOLO)…")
        for name, img_bytes, label_text in parse_correct(CORRECT):
            stem = Path(name).stem
            if stem in seen:
                n_dup += 1
                continue
            seen.add(stem)
            out.writestr(f"obj_train_data/{stem}.png", img_bytes)
            out.writestr(f"obj_train_data/{stem}.txt", label_text)
            image_paths.append(f"obj_train_data/{stem}.png")
            n_correct += 1

        out.writestr("train.txt", "\n".join(image_paths))

    print()
    print(f"From problematic_reviewed : {n_problematic}")
    print(f"From correct              : {n_correct}")
    print(f"Duplicates skipped        : {n_dup}")
    print(f"Total images in merged    : {len(image_paths)}")
    print(f"\nDone → {OUT_ZIP}")


if __name__ == "__main__":
    main()
