#!/usr/bin/env python3
"""Run a YOLO checkpoint on sample images and save annotated predictions."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_MODEL = "runs/yolo26/txl_pbc_yolo26m2/weights/best.pt"
FALLBACK_MODEL = "runs/yolo26/txl_pbc_yolo26m/weights/best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create annotated prediction samples from a YOLO model.")
    parser.add_argument("--model", default=None, help="Checkpoint path. Defaults to latest known best.pt.")
    parser.add_argument("--data", default="runs/datasets/txl_pbc_preprocessed/data.yaml", help="Dataset data.yaml.")
    parser.add_argument("--source", default=None, help="Image file or directory. Defaults to data.yaml test split.")
    parser.add_argument("--output", default="runs/prediction_samples", help="Output directory.")
    parser.add_argument("--num-samples", type=int, default=12, help="Number of images to sample from a directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 0,1, etc.")
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--show-ground-truth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def import_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install with: python3 -m pip install -U -r requirements.txt"
        ) from exc
    return YOLO


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def normalise_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def resolve_model(path_value: str | None) -> Path:
    if path_value:
        model = Path(path_value).expanduser()
    elif Path(DEFAULT_MODEL).exists():
        model = Path(DEFAULT_MODEL)
    elif Path(FALLBACK_MODEL).exists():
        model = Path(FALLBACK_MODEL)
    else:
        candidates = sorted(Path("runs").glob("**/weights/best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
        model = candidates[0] if candidates else Path(DEFAULT_MODEL)
    if not model.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model}")
    return model.resolve()


def resolve_dataset_path(data_yaml: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    data = load_yaml(data_yaml)
    root = Path(data.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return (root / path).resolve()


def resolve_source(args: argparse.Namespace, data_yaml: Path) -> Path:
    if args.source:
        return Path(args.source).expanduser().resolve()
    data = load_yaml(data_yaml)
    source = resolve_dataset_path(data_yaml, data.get("test") or data.get("val"))
    if source is None:
        raise ValueError("Could not resolve source images from --source or data.yaml test/val split")
    return source


def collect_images(source: Path, num_samples: int, seed: int) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    images = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images found in {source}")
    if num_samples > 0 and len(images) > num_samples:
        rng = random.Random(seed)
        images = sorted(rng.sample(images, num_samples))
    return images


def default_font(size: int = 14) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (22, 163, 74),
        (220, 38, 38),
        (37, 99, 235),
        (202, 138, 4),
        (147, 51, 234),
        (14, 165, 233),
    ]
    return palette[class_id % len(palette)]


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
    line_width: int,
    font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
    text_w, text_h = text_size(draw, label, font)
    pad = 3
    y_text = max(0, y1 - text_h - 2 * pad)
    draw.rectangle((x1, y_text, x1 + text_w + 2 * pad, y_text + text_h + 2 * pad), fill=color)
    draw.text((x1 + pad, y_text + pad), label, fill=(255, 255, 255), font=font)


def label_path_for_image(image_path: Path, data_yaml: Path) -> Path | None:
    data = load_yaml(data_yaml)
    root = Path(data.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    try:
        relative = image_path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    parts = list(relative.parts)
    if "images" not in parts:
        return None
    parts[parts.index("images")] = "labels"
    return (root / Path(*parts)).with_suffix(".txt")


def draw_ground_truth(image: Image.Image, image_path: Path, data_yaml: Path, names: list[str], line_width: int) -> None:
    label_path = label_path_for_image(image_path, data_yaml)
    if not label_path or not label_path.exists():
        return
    draw = ImageDraw.Draw(image)
    font = default_font(13)
    width, height = image.size
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        x_center, y_center, box_w, box_h = (float(value) for value in parts[1:])
        x1 = (x_center - box_w / 2) * width
        y1 = (y_center - box_h / 2) * height
        x2 = (x_center + box_w / 2) * width
        y2 = (y_center + box_h / 2) * height
        name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
        draw_box(draw, (x1, y1, x2, y2), f"GT {name}", (107, 114, 128), line_width, font)


def annotate_prediction(
    result: Any,
    image_path: Path,
    output_path: Path,
    names: list[str],
    data_yaml: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    if args.show_ground_truth:
        draw_ground_truth(image, image_path, data_yaml, names, args.line_width)
    draw = ImageDraw.Draw(image)
    font = default_font(14)
    detections: list[dict[str, Any]] = []

    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        for box, conf, class_value in zip(xyxy, confs, classes, strict=False):
            class_id = int(class_value)
            name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
            label = f"{name} {conf:.2f}"
            draw_box(draw, tuple(box), label, color_for_class(class_id), args.line_width, font)
            detections.append(
                {
                    "image": str(image_path),
                    "class_id": class_id,
                    "class_name": name,
                    "confidence": round(float(conf), 6),
                    "x1": round(float(box[0]), 2),
                    "y1": round(float(box[1]), 2),
                    "x2": round(float(box[2]), 2),
                    "y2": round(float(box[3]), 2),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return detections


def save_detection_tables(output_dir: Path, detections: list[dict[str, Any]]) -> None:
    json_path = output_dir / "predictions.json"
    json_path.write_text(json.dumps(detections, indent=2), encoding="utf-8")

    csv_path = output_dir / "predictions.csv"
    fieldnames = ["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detections)


def main() -> int:
    args = parse_args()
    data_yaml = Path(args.data).expanduser().resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    data = load_yaml(data_yaml)
    names = normalise_names(data.get("names"))
    model_path = resolve_model(args.model)
    source = resolve_source(args, data_yaml)
    images = collect_images(source, args.num_samples, args.seed)
    output_dir = Path(args.output).expanduser().resolve()
    if output_dir.exists() and not args.exist_ok:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    YOLO = import_yolo()
    model = YOLO(str(model_path))
    predict_args: dict[str, Any] = {
        "source": [str(path) for path in images],
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "save": False,
        "verbose": False,
    }
    if args.device != "auto":
        predict_args["device"] = args.device
    results = model.predict(**predict_args)

    all_detections: list[dict[str, Any]] = []
    for result, image_path in zip(results, images, strict=True):
        output_path = output_dir / f"{image_path.stem}_pred.jpg"
        all_detections.extend(annotate_prediction(result, image_path, output_path, names, data_yaml, args))
        if args.save_crops:
            crop_dir = output_dir / "crops" / image_path.stem
            crop_dir.mkdir(parents=True, exist_ok=True)
            source_result = getattr(result, "save_crop", None)
            if callable(source_result):
                source_result(save_dir=str(crop_dir), file_name=image_path.stem)

    save_detection_tables(output_dir, all_detections)
    shutil.copy2(data_yaml, output_dir / "data.yaml")
    summary = {
        "model": str(model_path),
        "source": str(source),
        "output": str(output_dir),
        "images": len(images),
        "detections": len(all_detections),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Prediction interrupted.", file=sys.stderr)
        raise SystemExit(130)
