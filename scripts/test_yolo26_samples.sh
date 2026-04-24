#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL=""
DATA="$ROOT_DIR/runs/datasets/txl_pbc_preprocessed/data.yaml"
SOURCE=""
OUTPUT="$ROOT_DIR/runs/prediction_samples"
NUM_SAMPLES="12"
SEED="42"
IMGSZ="640"
CONF="0.25"
IOU="0.7"
DEVICE="auto"
LINE_WIDTH="2"
EXIST_OK_ARG=(--exist-ok)
GROUND_TRUTH_ARG=(--no-show-ground-truth)
CROPS_ARG=(--no-save-crops)
DRY_RUN="0"

usage() {
  cat <<'USAGE'
Create annotated YOLO26 prediction samples.

Usage:
  scripts/test_yolo26_samples.sh [options]

Options:
  --model FILE          Model checkpoint; defaults to latest known best.pt
  --data FILE           Dataset data.yaml (default: runs/datasets/txl_pbc_preprocessed/data.yaml)
  --source PATH         Image file or image directory; defaults to data.yaml test split
  --output DIR          Output directory (default: runs/prediction_samples)
  --num-samples N       Number of images sampled from a directory (default: 12)
  --seed N              Random sample seed (default: 42)
  --imgsz N             Inference image size (default: 640)
  --conf FLOAT          Confidence threshold (default: 0.25)
  --iou FLOAT           NMS IoU threshold (default: 0.7)
  --device DEVICE       auto, cpu, 0, 0,1, etc. (default: auto)
  --line-width N        Box line width (default: 2)
  --show-ground-truth   Also draw gray GT boxes from matching YOLO labels
  --save-crops          Save detected crops when supported by Ultralytics
  --no-exist-ok         Fail if output directory already exists
  --dry-run             Print command without executing it
  -h, --help            Show this help

Example:
  scripts/test_yolo26_samples.sh --model runs/yolo26/txl_pbc_yolo26m2/weights/best.pt --device 0 --num-samples 20
USAGE
}

require_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

print_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      require_value "$@"; MODEL="$2"; shift 2 ;;
    --data)
      require_value "$@"; DATA="$2"; shift 2 ;;
    --source)
      require_value "$@"; SOURCE="$2"; shift 2 ;;
    --output)
      require_value "$@"; OUTPUT="$2"; shift 2 ;;
    --num-samples)
      require_value "$@"; NUM_SAMPLES="$2"; shift 2 ;;
    --seed)
      require_value "$@"; SEED="$2"; shift 2 ;;
    --imgsz)
      require_value "$@"; IMGSZ="$2"; shift 2 ;;
    --conf)
      require_value "$@"; CONF="$2"; shift 2 ;;
    --iou)
      require_value "$@"; IOU="$2"; shift 2 ;;
    --device)
      require_value "$@"; DEVICE="$2"; shift 2 ;;
    --line-width)
      require_value "$@"; LINE_WIDTH="$2"; shift 2 ;;
    --show-ground-truth)
      GROUND_TRUTH_ARG=(--show-ground-truth); shift ;;
    --save-crops)
      CROPS_ARG=(--save-crops); shift ;;
    --no-exist-ok)
      EXIST_OK_ARG=(--no-exist-ok); shift ;;
    --dry-run)
      DRY_RUN="1"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

CMD=(
  python3 "$ROOT_DIR/visualize_yolo_predictions.py"
  --data "$DATA"
  --output "$OUTPUT"
  --num-samples "$NUM_SAMPLES"
  --seed "$SEED"
  --imgsz "$IMGSZ"
  --conf "$CONF"
  --iou "$IOU"
  --device "$DEVICE"
  --line-width "$LINE_WIDTH"
  "${EXIST_OK_ARG[@]}"
  "${GROUND_TRUTH_ARG[@]}"
  "${CROPS_ARG[@]}"
)

if [[ -n "$MODEL" ]]; then
  CMD+=(--model "$MODEL")
fi
if [[ -n "$SOURCE" ]]; then
  CMD+=(--source "$SOURCE")
fi

print_command "${CMD[@]}"
if [[ "$DRY_RUN" == "0" ]]; then
  "${CMD[@]}"
fi
