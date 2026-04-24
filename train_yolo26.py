#!/usr/bin/env python3
"""Fine-tune an Ultralytics YOLO26 detection model."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml


MODEL_SIZES = ("n", "s", "m", "l", "x")
ULTRALYTICS_ASSETS_LATEST = "https://github.com/ultralytics/assets/releases/latest/download"
MIN_YOLO26_ULTRALYTICS = (8, 4, 0)


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_key_value(items: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE for --set, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if not key:
            raise ValueError(f"Empty key in --set argument: {item}")
        parsed[key] = parse_scalar(value.strip())
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26 on a YOLO-format dataset.")
    parser.add_argument("--data", default="runs/datasets/txl_pbc_preprocessed/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--model-size", choices=MODEL_SIZES, default="n", help="YOLO26 model size.")
    parser.add_argument("--model", default=None, help="Optional explicit model/checkpoint path.")
    parser.add_argument("--resume", default=None, help="Resume from a previous last.pt checkpoint.")
    parser.add_argument("--weights-dir", default="runs/weights", help="Directory for auto-downloaded base weights.")
    parser.add_argument(
        "--download-base-url",
        default=ULTRALYTICS_ASSETS_LATEST,
        help="Base URL used when auto-downloading missing YOLO26 weights.",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default=16, help="Batch size, or -1 for Ultralytics auto batch.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, 0, 0,1, etc.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--close-mosaic", type=int, default=10)

    parser.add_argument("--cache", default="false", help="false, true, ram, or disk.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--project", default="runs/yolo26")
    parser.add_argument("--name", default=None)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--final-val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-split", choices=("val", "test"), default="val")
    parser.add_argument("--pretrain-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrain-eval-split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--history-dirname",
        default="training_history",
        help="Directory name inside the run folder for append-only training history.",
    )
    parser.add_argument(
        "--metadata-dirname",
        default="deployment_metadata",
        help="Directory name inside the run folder for self-contained metadata.",
    )
    parser.add_argument(
        "--set",
        dest="extra",
        action="append",
        default=[],
        help="Extra Ultralytics train arg as KEY=VALUE. Can be repeated.",
    )
    return parser.parse_args()


def import_yolo():
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics.\n"
            "Install the current Ultralytics package with:\n"
            "  python3 -m pip install -U ultralytics"
        ) from exc
    version = parse_version(getattr(ultralytics, "__version__", "0"))
    if version < MIN_YOLO26_ULTRALYTICS:
        required = ".".join(str(part) for part in MIN_YOLO26_ULTRALYTICS)
        raise SystemExit(
            f"Ultralytics {ultralytics.__version__} is too old for YOLO26 weights.\n"
            f"Install Ultralytics >= {required}:\n"
            "  python3 -m pip install -U 'ultralytics>=8.4.0'"
        )
    return YOLO


def parse_version(version: str) -> tuple[int, int, int]:
    parts: list[int] = []
    for raw_part in version.split(".")[:3]:
        digits = "".join(char for char in raw_part if char.isdigit())
        parts.append(int(digits or 0))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def infer_model_size_from_path(path_value: str) -> str | None:
    match = re.search(r"yolo26([nsmxl])", path_value)
    return match.group(1) if match else None


def normalise_batch(value: str) -> int | float:
    parsed = parse_scalar(str(value))
    if not isinstance(parsed, (int, float)):
        raise ValueError(f"--batch must be numeric, got {value!r}")
    return parsed


def resolve_model(args: argparse.Namespace) -> str:
    if args.resume:
        resume_path = Path(args.resume).expanduser()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        return str(resume_path)
    if args.model:
        return args.model
    return f"yolo26{args.model_size}.pt"


def is_yolo26_weight_name(model_ref: str) -> bool:
    name = Path(model_ref).name
    return name in {f"yolo26{size}.pt" for size in MODEL_SIZES}


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    if temporary.exists():
        temporary.unlink()

    print(f"Downloading missing model weights: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
    except urllib.error.URLError as exc:
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(
            f"Could not download {url}. Check internet access or pass --model /path/to/weights.pt."
        ) from exc
    temporary.replace(destination)


def ensure_model_available(model_ref: str, args: argparse.Namespace) -> str:
    path = Path(model_ref).expanduser()
    if path.exists():
        return str(path)
    if path.parent != Path("."):
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    if not is_yolo26_weight_name(model_ref):
        return model_ref

    weights_dir = Path(args.weights_dir).expanduser()
    destination = weights_dir / Path(model_ref).name
    if destination.exists():
        return str(destination)

    base_url = args.download_base_url.rstrip("/")
    download_file(f"{base_url}/{Path(model_ref).name}", destination)
    return str(destination)


def build_train_args(args: argparse.Namespace) -> dict[str, Any]:
    data_path = Path(args.data).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    run_name = args.name or f"txl_pbc_yolo26{args.model_size}"
    train_args: dict[str, Any] = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": normalise_batch(args.batch),
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "cos_lr": args.cos_lr,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "fliplr": args.fliplr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "close_mosaic": args.close_mosaic,
        "cache": parse_scalar(str(args.cache)),
        "amp": args.amp,
        "plots": args.plots,
        "val": args.val,
        "save": args.save,
        "save_period": args.save_period,
        "project": args.project,
        "name": run_name,
        "exist_ok": args.exist_ok,
    }
    if args.device != "auto":
        train_args["device"] = args.device
    if args.resume:
        train_args["resume"] = True

    train_args.update(parse_key_value(args.extra))
    return train_args


def print_config(model_ref: str, train_args: dict[str, Any]) -> None:
    printable = {
        "model": model_ref,
        **train_args,
    }
    print("Training configuration:")
    print(json.dumps(printable, indent=2, sort_keys=True))


def run_command(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def collect_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("ultralytics", "torch", "torchvision", "numpy", "PIL", "yaml"):
        try:
            module = __import__(package)
        except ImportError:
            versions[package] = None
            continue
        versions[package] = getattr(module, "__version__", None)
    return versions


def detect_cuda() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}

    cuda_info: dict[str, Any] = {"available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        cuda_info["device_count"] = torch.cuda.device_count()
        cuda_info["devices"] = [
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "capability": ".".join(str(part) for part in torch.cuda.get_device_capability(idx)),
            }
            for idx in range(torch.cuda.device_count())
        ]
    return cuda_info


def get_save_dir(results: Any, model: Any, train_args: dict[str, Any]) -> Path:
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None and getattr(model, "trainer", None) is not None:
        save_dir = getattr(model.trainer, "save_dir", None)
    if save_dir is None:
        save_dir = Path(train_args["project"]) / str(train_args["name"])
    return Path(save_dir)


def checkpoint_paths(save_dir: Path) -> dict[str, str | None]:
    weights_dir = save_dir / "weights"
    paths = {
        "best": weights_dir / "best.pt",
        "last": weights_dir / "last.pt",
    }
    return {key: str(path) if path.exists() else None for key, path in paths.items()}


def json_safe(value: Any) -> Any:
    if value is None:
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int) and type(value) is not bool:
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(row), sort_keys=True) + "\n")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_row = flatten_dict(json_safe(row))
    existing_fields: list[str] = []
    existing_rows: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)

    fields = list(existing_fields)
    for key in flat_row:
        if key not in fields:
            fields.append(key)

    rewrite = fields != existing_fields and path.exists() and path.stat().st_size > 0
    mode = "w" if rewrite else "a"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if rewrite or not existing_fields:
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow({key: existing_row.get(key, "") for key in fields})
        writer.writerow({key: flat_row.get(key, "") for key in fields})


def history_dir_for_save_dir(save_dir: Path, args: argparse.Namespace) -> Path:
    return save_dir / args.history_dirname


def increment_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.name}{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find available run directory for {path}")


def predicted_save_dir(args: argparse.Namespace, train_args: dict[str, Any]) -> Path:
    if args.resume:
        return Path(args.resume).expanduser().resolve().parents[1]
    base = Path(str(train_args["project"])).expanduser() / str(train_args["name"])
    if train_args.get("exist_ok"):
        return base
    return increment_path(base)


def record_history_event(save_dir: Path, args: argparse.Namespace, row: dict[str, Any]) -> None:
    history_dir = history_dir_for_save_dir(save_dir, args)
    append_jsonl(history_dir / "history.jsonl", row)
    append_csv(history_dir / "history.csv", row)


def metrics_to_dict(metrics: Any) -> dict[str, Any]:
    if metrics is None:
        return {}
    result: dict[str, Any] = {}
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        result.update(results_dict)
    for attr in ("fitness", "speed", "maps", "names"):
        if hasattr(metrics, attr):
            value = getattr(metrics, attr)
            result[attr] = value() if callable(value) and attr == "fitness" else value
    box = getattr(metrics, "box", None)
    if box is not None:
        for attr in ("mp", "mr", "map50", "map", "maps"):
            if hasattr(box, attr):
                result[f"box.{attr}"] = getattr(box, attr)
    return json_safe(result)


def run_pretrain_eval(model: Any, args: argparse.Namespace, train_args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    val_args: dict[str, Any] = {
        "data": train_args["data"],
        "split": args.pretrain_eval_split,
        "imgsz": train_args["imgsz"],
        "batch": train_args["batch"],
        "plots": False,
    }
    if args.device != "auto":
        val_args["device"] = args.device
    print(f"Running pre-training baseline evaluation on split={args.pretrain_eval_split}...")
    metrics = model.val(**val_args)
    metrics_dict = metrics_to_dict(metrics)
    print("Pre-training baseline metrics:")
    print(json.dumps(metrics_dict, indent=2, sort_keys=True))
    return metrics, metrics_dict


def write_history_config(save_dir: Path, args: argparse.Namespace, model_ref: str, train_args: dict[str, Any]) -> None:
    history_dir = history_dir_for_save_dir(save_dir, args)
    history_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "model": model_ref,
        "resume_from": str(Path(args.resume).resolve()) if args.resume else None,
        "train_args": train_args,
        "pretrain_eval": args.pretrain_eval,
        "pretrain_eval_split": args.pretrain_eval_split,
    }
    (history_dir / "run_config.json").write_text(json.dumps(json_safe(config), indent=2), encoding="utf-8")


def add_history_callbacks(model: Any, args: argparse.Namespace) -> None:
    if not hasattr(model, "add_callback"):
        return

    seen_epochs: set[int] = set()

    def on_fit_epoch_end(trainer: Any) -> None:
        save_dir = Path(getattr(trainer, "save_dir", Path(".")))
        epoch = int(getattr(trainer, "epoch", -1)) + 1
        if epoch in seen_epochs:
            return
        seen_epochs.add(epoch)
        row = {
            "event": "epoch_end",
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "epoch": epoch,
            "metrics": json_safe(getattr(trainer, "metrics", {})),
            "train_loss": json_safe(getattr(trainer, "tloss", None)),
            "learning_rate": json_safe(getattr(trainer, "lr", None)),
        }
        record_history_event(save_dir, args, row)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def write_model_card(metadata_dir: Path, metadata: dict[str, Any]) -> Path:
    checkpoints = metadata["artifacts"]["checkpoints"]
    lines = [
        "# YOLO26 TXL-PBC Model Card",
        "",
        f"- Base model: `{metadata['model']['base_model']}`",
        f"- Model size: `{metadata['model']['model_size']}`",
        f"- Dataset YAML: `{metadata['dataset']['data_yaml']}`",
        f"- Classes: {', '.join(metadata['dataset']['names'])}",
        f"- Best checkpoint: `{checkpoints.get('best')}`",
        f"- Last checkpoint: `{checkpoints.get('last')}`",
        "",
        "## Rebuild Training Run",
        "",
        "```bash",
        metadata["reproducibility"]["command"],
        "```",
        "",
        "## Run Inference",
        "",
        "```python",
        "from ultralytics import YOLO",
        f"model = YOLO({checkpoints.get('best')!r})",
        "results = model.predict('path/to/image.png', imgsz="
        f"{metadata['training_args']['imgsz']}, conf=0.25)",
        "```",
    ]
    model_card = metadata_dir / "MODEL_CARD.md"
    model_card.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return model_card


def write_metadata_package(
    args: argparse.Namespace,
    model_ref: str,
    train_args: dict[str, Any],
    save_dir: Path,
    metrics: Any,
    pretrain_metrics: dict[str, Any] | None,
) -> Path:
    data_yaml = Path(train_args["data"]).resolve()
    dataset = load_yaml(data_yaml)
    names = normalise_names(dataset.get("names"))
    metadata_dir = save_dir / args.metadata_dirname
    metadata_dir.mkdir(parents=True, exist_ok=True)

    copied_data_yaml = metadata_dir / "data.yaml"
    shutil.copy2(data_yaml, copied_data_yaml)
    preprocess_report = data_yaml.parent / "preprocess_report.json"
    if preprocess_report.exists():
        shutil.copy2(preprocess_report, metadata_dir / preprocess_report.name)

    ckpts = checkpoint_paths(save_dir)
    metadata: dict[str, Any] = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "task": "detect",
        "framework": "ultralytics",
        "model": {
            "base_model": model_ref,
            "model_size": args.model_size,
            "resume_from": str(Path(args.resume).resolve()) if args.resume else None,
        },
        "dataset": {
            "data_yaml": str(copied_data_yaml),
            "source_data_yaml": str(data_yaml),
            "nc": int(dataset.get("nc", len(names)) or len(names)),
            "names": names,
            "path": dataset.get("path"),
            "train": dataset.get("train"),
            "val": dataset.get("val"),
            "test": dataset.get("test"),
        },
        "training_args": train_args,
        "pretraining_baseline": pretrain_metrics,
        "artifacts": {
            "run_dir": str(save_dir),
            "metadata_dir": str(metadata_dir),
            "history_dir": str(history_dir_for_save_dir(save_dir, args)),
            "checkpoints": ckpts,
            "checkpoint_sha256": {},
        },
        "environment": {
            "platform": platform.platform(),
            "packages": collect_package_versions(),
            "cuda": detect_cuda(),
        },
        "reproducibility": {
            "command": " ".join(shlex_quote(arg) for arg in [sys.executable, *sys.argv]),
            "git_commit": run_command(["git", "rev-parse", "HEAD"]),
            "git_status_short": run_command(["git", "status", "--short"]),
        },
        "final_validation": str(metrics) if metrics is not None else None,
    }
    metadata = json_safe(metadata)

    embed_metadata_in_checkpoints(ckpts, metadata)
    metadata["artifacts"]["checkpoint_sha256"] = {
        key: sha256_file(Path(path)) if path else None for key, path in ckpts.items()
    }
    metadata_json = metadata_dir / "metadata.json"
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metadata_yaml = metadata_dir / "metadata.yaml"
    metadata_yaml.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    write_model_card(metadata_dir, metadata)
    return metadata_dir


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def embed_metadata_in_checkpoints(ckpts: dict[str, str | None], metadata: dict[str, Any]) -> None:
    try:
        import torch
    except ImportError:
        return

    embedded = {
        "task": metadata["task"],
        "framework": metadata["framework"],
        "model": metadata["model"],
        "dataset": metadata["dataset"],
        "training_args": metadata["training_args"],
        "pretraining_baseline": metadata.get("pretraining_baseline"),
        "environment": metadata["environment"],
        "reproducibility": metadata["reproducibility"],
    }
    for path_value in ckpts.values():
        if not path_value:
            continue
        path = Path(path_value)
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict):
                checkpoint["txl_pbc_metadata"] = embedded
                torch.save(checkpoint, path)
        except Exception as exc:  # noqa: BLE001 - sidecar metadata remains authoritative
            print(f"Could not embed metadata into {path}: {exc}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    inferred_size = infer_model_size_from_path(args.resume or args.model or "")
    if inferred_size and args.model_size == "n":
        args.model_size = inferred_size
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be > 0")

    YOLO = import_yolo()
    model_ref = resolve_model(args)
    model_ref = ensure_model_available(model_ref, args)
    train_args = build_train_args(args)
    initial_save_dir = predicted_save_dir(args, train_args)
    if not args.resume and not train_args.get("exist_ok"):
        train_args["name"] = initial_save_dir.name
        train_args["project"] = str(initial_save_dir.parent)
        train_args["exist_ok"] = True
    print_config(model_ref, train_args)

    model = YOLO(model_ref)
    pretrain_metrics_dict: dict[str, Any] | None = None
    if args.pretrain_eval:
        _, pretrain_metrics_dict = run_pretrain_eval(model, args, train_args)
    write_history_config(initial_save_dir, args, model_ref, train_args)
    if pretrain_metrics_dict is not None:
        record_history_event(
            initial_save_dir,
            args,
            {
                "event": "pretrain_eval",
                "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                "epoch": 0,
                "split": args.pretrain_eval_split,
                "resume_from": str(Path(args.resume).resolve()) if args.resume else None,
                "metrics": pretrain_metrics_dict,
                "note": "Object detection uses precision/recall/mAP as accuracy-like metrics. Loss fields are present only if Ultralytics exposes them during standalone validation.",
            },
        )
        history_dir = history_dir_for_save_dir(initial_save_dir, args)
        (history_dir / "pretrain_metrics.json").write_text(
            json.dumps(json_safe(pretrain_metrics_dict), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    add_history_callbacks(model, args)
    results = model.train(**train_args)

    save_dir = get_save_dir(results, model, train_args)
    print(f"Training artifacts: {save_dir}")
    if save_dir.resolve() != initial_save_dir.resolve():
        source_history = history_dir_for_save_dir(initial_save_dir, args)
        target_history = history_dir_for_save_dir(save_dir, args)
        if source_history.exists():
            shutil.copytree(source_history, target_history, dirs_exist_ok=True)
    write_history_config(save_dir, args, model_ref, train_args)
    if pretrain_metrics_dict is not None:
        history_dir = history_dir_for_save_dir(save_dir, args)
        (history_dir / "pretrain_metrics.json").write_text(
            json.dumps(json_safe(pretrain_metrics_dict), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    metrics = None
    if args.final_val:
        val_args: dict[str, Any] = {
            "data": train_args["data"],
            "split": args.val_split,
            "imgsz": train_args["imgsz"],
            "batch": train_args["batch"],
            "plots": args.plots,
        }
        if args.device != "auto":
            val_args["device"] = args.device
        metrics = model.val(**val_args)
        print(f"Final {args.val_split} metrics: {metrics}")
    if args.save:
        metadata_dir = write_metadata_package(args, model_ref, train_args, save_dir, metrics, pretrain_metrics_dict)
        print(f"Deployment metadata: {metadata_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Training interrupted.", file=sys.stderr)
        raise SystemExit(130)
