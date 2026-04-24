#!/usr/bin/env python3
"""Validate and prepare a YOLO-format dataset for Ultralytics training."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit("Missing dependency: PyYAML. Install with: python3 -m pip install pyyaml") from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit("Missing dependency: Pillow. Install with: python3 -m pip install pillow") from exc


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass
class SplitReport:
    images: int = 0
    labels: int = 0
    annotations_in: int = 0
    annotations_out: int = 0
    skipped_annotations: int = 0
    fixed_annotations: int = 0
    missing_labels: list[str] = field(default_factory=list)
    missing_images: list[str] = field(default_factory=list)
    invalid_images: list[str] = field(default_factory=list)
    class_counts: Counter[int] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate image/label pairs, clean YOLO labels, optionally resize images, "
            "and write a data.yaml with absolute dataset paths."
        )
    )
    parser.add_argument("--source", default="TXL-PBC", help="Source dataset directory.")
    parser.add_argument("--output", default="runs/datasets/txl_pbc_preprocessed", help="Prepared dataset directory.")
    parser.add_argument("--classes-file", default=None, help="Optional classes.txt path.")
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to place images in the prepared dataset when no resizing is requested.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        help="Resize images so the longest side is at most this value. Use 0 to keep originals.",
    )
    parser.add_argument(
        "--no-clip-labels",
        action="store_true",
        help="Fail invalid/out-of-bounds labels instead of clipping boxes to image bounds.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not write output files.")
    parser.add_argument("--report-name", default="preprocess_report.json", help="Report filename inside output.")
    return parser.parse_args()


def load_dataset_yaml(source: Path) -> dict[str, Any]:
    yaml_path = source / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing dataset YAML: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {yaml_path}")
    return data


def load_class_names(source: Path, classes_file: str | None, dataset_yaml: dict[str, Any]) -> list[str]:
    names = dataset_yaml.get("names")
    if isinstance(names, dict):
        ordered = [names[key] for key in sorted(names, key=lambda item: int(item))]
        return [str(name) for name in ordered]
    if isinstance(names, list) and names:
        return [str(name) for name in names]

    class_path = Path(classes_file) if classes_file else source / "classes.txt"
    if not class_path.exists():
        nc = int(dataset_yaml.get("nc", 0) or 0)
        if nc <= 0:
            raise ValueError("Could not determine class names from data.yaml or classes.txt")
        return [f"class_{idx}" for idx in range(nc)]

    class_names: list[str] = []
    for raw_line in class_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if parts[0].isdigit() and len(parts) == 2:
            class_names.append(parts[1])
        else:
            class_names.append(line)
    if not class_names:
        raise ValueError(f"No classes found in {class_path}")
    return class_names


def image_files_by_stem(directory: Path) -> tuple[dict[str, Path], list[str]]:
    images: dict[str, Path] = {}
    duplicates: list[str] = []
    if not directory.exists():
        return images, duplicates
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in images:
            duplicates.append(path.stem)
            continue
        images[path.stem] = path
    return images, duplicates


def label_files_by_stem(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        return {}
    return {path.stem: path for path in sorted(directory.glob("*.txt")) if path.is_file()}


def open_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def parse_label_line(
    line: str,
    class_count: int,
    clip_labels: bool,
) -> tuple[str | None, int | None, bool]:
    parts = line.split()
    if len(parts) != 5:
        return None, None, False

    try:
        class_value = float(parts[0])
        x_center, y_center, width, height = (float(value) for value in parts[1:])
    except ValueError:
        return None, None, False

    values = (x_center, y_center, width, height)
    if not class_value.is_integer():
        return None, None, False
    class_id = int(class_value)
    if class_id < 0 or class_id >= class_count or not all(math.isfinite(value) for value in values):
        return None, None, False
    if width <= 0 or height <= 0:
        return None, None, False

    left = x_center - width / 2
    top = y_center - height / 2
    right = x_center + width / 2
    bottom = y_center + height / 2
    original_box = (left, top, right, bottom)

    if not clip_labels and not all(0.0 <= value <= 1.0 for value in original_box):
        return None, None, False

    left = max(0.0, min(1.0, left))
    top = max(0.0, min(1.0, top))
    right = max(0.0, min(1.0, right))
    bottom = max(0.0, min(1.0, bottom))
    if right <= left or bottom <= top:
        return None, None, False

    clipped = original_box != (left, top, right, bottom)
    x_center = (left + right) / 2
    y_center = (top + bottom) / 2
    width = right - left
    height = bottom - top
    cleaned = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    return cleaned, class_id, clipped


def prepare_image(src: Path, dst: Path, image_size: int, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if image_size > 0:
        with Image.open(src) as image:
            image = image.convert("RGB") if src.suffix.lower() in {".jpg", ".jpeg"} else image.copy()
            image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
            image.save(dst)
        return

    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_mode == "symlink":
        dst.symlink_to(src.resolve())
    elif copy_mode == "hardlink":
        try:
            dst.hardlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def process_split(
    split: str,
    source: Path,
    output: Path,
    class_count: int,
    args: argparse.Namespace,
) -> tuple[SplitReport, list[str]]:
    report = SplitReport()
    warnings: list[str] = []
    image_dir = source / "images" / split
    label_dir = source / "labels" / split
    images, duplicate_image_stems = image_files_by_stem(image_dir)
    labels = label_files_by_stem(label_dir)

    report.images = len(images)
    report.labels = len(labels)
    if duplicate_image_stems:
        warnings.append(f"{split}: duplicate image stems ignored: {duplicate_image_stems[:10]}")

    report.missing_labels = sorted(set(images) - set(labels))
    report.missing_images = sorted(set(labels) - set(images))

    out_image_dir = output / "images" / split
    out_label_dir = output / "labels" / split
    if not args.dry_run:
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

    for stem, image_path in images.items():
        try:
            open_image(image_path)
        except Exception as exc:  # noqa: BLE001 - reported to user, then skipped
            report.invalid_images.append(str(image_path))
            warnings.append(f"{split}/{image_path.name}: invalid image skipped ({exc})")
            continue

        label_path = labels.get(stem)
        cleaned_lines: list[str] = []
        if label_path and label_path.exists():
            for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                report.annotations_in += 1
                cleaned, class_id, fixed = parse_label_line(stripped, class_count, not args.no_clip_labels)
                if cleaned is None or class_id is None:
                    report.skipped_annotations += 1
                    warnings.append(f"{split}/{label_path.name}:{line_number}: invalid annotation skipped")
                    continue
                cleaned_lines.append(cleaned)
                report.class_counts[class_id] += 1
                report.annotations_out += 1
                if fixed:
                    report.fixed_annotations += 1

        if args.dry_run:
            continue

        prepare_image(image_path, out_image_dir / image_path.name, args.image_size, args.copy_mode)
        (out_label_dir / f"{stem}.txt").write_text("\n".join(cleaned_lines) + ("\n" if cleaned_lines else ""), encoding="utf-8")

    return report, warnings


def write_data_yaml(output: Path, class_names: list[str]) -> Path:
    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    data_yaml = output / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_yaml


def serialise_report(report: dict[str, Any]) -> dict[str, Any]:
    serialised = dict(report)
    split_reports: dict[str, Any] = {}
    for split, split_report in serialised["splits"].items():
        item = split_report.__dict__.copy()
        item["class_counts"] = {str(key): value for key, value in sorted(item["class_counts"].items())}
        split_reports[split] = item
    serialised["splits"] = split_reports
    return serialised


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()

    if args.image_size < 0:
        raise ValueError("--image-size must be >= 0")
    if source == output:
        raise ValueError("--output must be different from --source")

    dataset_yaml = load_dataset_yaml(source)
    class_names = load_class_names(source, args.classes_file, dataset_yaml)
    yaml_nc = int(dataset_yaml.get("nc", len(class_names)) or len(class_names))
    if yaml_nc != len(class_names):
        raise ValueError(f"data.yaml nc={yaml_nc}, but {len(class_names)} class names were found")

    full_report: dict[str, Any] = {
        "source": str(source),
        "output": str(output),
        "class_names": class_names,
        "splits": {},
        "warnings": [],
    }
    total_class_counts: Counter[int] = Counter()
    totals = defaultdict(int)

    for split in SPLITS:
        split_report, split_warnings = process_split(split, source, output, len(class_names), args)
        full_report["splits"][split] = split_report
        full_report["warnings"].extend(split_warnings)
        total_class_counts.update(split_report.class_counts)
        for key in ("images", "labels", "annotations_in", "annotations_out", "skipped_annotations", "fixed_annotations"):
            totals[key] += getattr(split_report, key)

    full_report["totals"] = dict(totals)
    full_report["total_class_counts"] = {
        class_names[idx]: total_class_counts.get(idx, 0) for idx in range(len(class_names))
    }

    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        data_yaml = write_data_yaml(output, class_names)
        (output / args.report_name).write_text(
            json.dumps(serialise_report(full_report), indent=2),
            encoding="utf-8",
        )
        print(f"Prepared dataset: {output}")
        print(f"YOLO data YAML: {data_yaml}")
    else:
        print("Dry run complete; no files were written.")

    print(
        "Summary: "
        f"{totals['images']} images, "
        f"{totals['annotations_out']} annotations kept, "
        f"{totals['fixed_annotations']} boxes clipped, "
        f"{totals['skipped_annotations']} annotations skipped."
    )
    if full_report["warnings"]:
        print(f"Warnings: {len(full_report['warnings'])} total. See report after a non-dry run for details.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
