#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SOURCE="$ROOT_DIR/TXL-PBC"
PREPROCESSED="$ROOT_DIR/runs/datasets/txl_pbc_preprocessed"
DATA_YAML=""
SIZE="n"
MODEL=""
RESUME=""
WEIGHTS_DIR="$ROOT_DIR/runs/weights"
EPOCHS="100"
IMGSZ="640"
BATCH="16"
DEVICE="auto"
WORKERS="8"
PATIENCE="50"
SEED="42"
OPTIMIZER="auto"
LR0="0.01"
LRF="0.01"
MOMENTUM="0.937"
WEIGHT_DECAY="0.0005"
WARMUP_EPOCHS="3.0"
HSV_H="0.015"
HSV_S="0.7"
HSV_V="0.4"
DEGREES="0.0"
TRANSLATE="0.1"
SCALE="0.5"
FLIPLR="0.5"
MOSAIC="1.0"
MIXUP="0.0"
CLOSE_MOSAIC="10"
CACHE="false"
SAVE_PERIOD="-1"
PROJECT="$ROOT_DIR/runs/yolo26"
NAME=""
HISTORY_DIRNAME="training_history"
PRETRAIN_EVAL_SPLIT="val"
COPY_MODE="copy"
PREPROCESS_IMAGE_SIZE="0"
PREPROCESS="1"
DRY_RUN="0"
EXIST_OK="0"
INSTALL_DEPS="0"

COS_LR_ARG=(--cos-lr)
AMP_ARG=(--amp)
PLOTS_ARG=(--plots)
VAL_ARG=(--val)
FINAL_VAL_ARG=(--final-val)
SAVE_ARG=(--save)
PRETRAIN_EVAL_ARG=(--pretrain-eval)
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Fine-tune YOLO26 on TXL-PBC.

Usage:
  scripts/train_yolo26.sh [options]

Common options:
  -s, --size n|s|m|l|x       YOLO26 size (default: n)
      --epochs N             Training epochs (default: 100)
      --imgsz N              Training image size (default: 640)
      --batch N              Batch size, or -1 for auto batch (default: 16)
      --device DEVICE        auto, cpu, 0, 0,1, etc. (default: auto)
      --model FILE           Start from an explicit model/checkpoint instead of yolo26<SIZE>.pt
      --resume FILE          Resume from a previous last.pt checkpoint
      --weights-dir DIR      Directory for auto-downloaded YOLO26 weights
      --workers N            Dataloader workers (default: 8)
      --lr0 FLOAT            Initial learning rate (default: 0.01)
      --lrf FLOAT            Final LR fraction (default: 0.01)
      --optimizer NAME       Optimizer passed to Ultralytics (default: auto)
      --patience N           Early-stopping patience (default: 50)
      --cache VALUE          false, true, ram, or disk (default: false)

Dataset options:
      --source DIR           Source YOLO dataset (default: TXL-PBC)
      --preprocessed DIR     Output prepared dataset dir (default: runs/datasets/txl_pbc_preprocessed)
      --data FILE            Use an existing data.yaml and skip preprocessing
      --no-preprocess        Skip preprocessing and use --preprocessed/data.yaml
      --copy-mode MODE       copy, hardlink, or symlink (default: copy)
      --preprocess-imgsz N   Resize longest image side during preprocessing; 0 keeps originals

Run options:
      --project DIR          Ultralytics project dir (default: runs/yolo26)
      --name NAME            Run name (default: txl_pbc_yolo26<SIZE>)
      --exist-ok             Allow overwriting the same run name
      --save-period N        Save checkpoint every N epochs; -1 disables periodic saves
      --history-dirname NAME Directory for append-only training history
      --pretrain-eval-split train|val|test  Split for epoch-0 baseline metrics
      --no-pretrain-eval    Skip epoch-0 baseline evaluation
      --no-save              Disable checkpoint saving
      --no-cos-lr            Disable cosine LR schedule
      --no-amp               Disable mixed precision
      --no-plots             Disable plot generation
      --no-val               Disable validation during training
      --no-final-val         Skip final validation after training
      --set KEY=VALUE        Pass any extra Ultralytics training argument; repeatable
      --install-deps         Install Python dependencies from requirements.txt before running
      --dry-run              Print commands without executing them
  -h, --help                 Show this help

Example:
  scripts/train_yolo26.sh --size s --epochs 150 --batch 8 --device 0 --lr0 0.005
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
    -s|--size)
      require_value "$@"; SIZE="$2"; shift 2 ;;
    --epochs)
      require_value "$@"; EPOCHS="$2"; shift 2 ;;
    --imgsz)
      require_value "$@"; IMGSZ="$2"; shift 2 ;;
    --batch)
      require_value "$@"; BATCH="$2"; shift 2 ;;
    --device)
      require_value "$@"; DEVICE="$2"; shift 2 ;;
    --model)
      require_value "$@"; MODEL="$2"; shift 2 ;;
    --resume)
      require_value "$@"; RESUME="$2"; PREPROCESS="0"; shift 2 ;;
    --weights-dir)
      require_value "$@"; WEIGHTS_DIR="$2"; shift 2 ;;
    --workers)
      require_value "$@"; WORKERS="$2"; shift 2 ;;
    --patience)
      require_value "$@"; PATIENCE="$2"; shift 2 ;;
    --seed)
      require_value "$@"; SEED="$2"; shift 2 ;;
    --optimizer)
      require_value "$@"; OPTIMIZER="$2"; shift 2 ;;
    --lr0)
      require_value "$@"; LR0="$2"; shift 2 ;;
    --lrf)
      require_value "$@"; LRF="$2"; shift 2 ;;
    --momentum)
      require_value "$@"; MOMENTUM="$2"; shift 2 ;;
    --weight-decay)
      require_value "$@"; WEIGHT_DECAY="$2"; shift 2 ;;
    --warmup-epochs)
      require_value "$@"; WARMUP_EPOCHS="$2"; shift 2 ;;
    --hsv-h)
      require_value "$@"; HSV_H="$2"; shift 2 ;;
    --hsv-s)
      require_value "$@"; HSV_S="$2"; shift 2 ;;
    --hsv-v)
      require_value "$@"; HSV_V="$2"; shift 2 ;;
    --degrees)
      require_value "$@"; DEGREES="$2"; shift 2 ;;
    --translate)
      require_value "$@"; TRANSLATE="$2"; shift 2 ;;
    --scale)
      require_value "$@"; SCALE="$2"; shift 2 ;;
    --fliplr)
      require_value "$@"; FLIPLR="$2"; shift 2 ;;
    --mosaic)
      require_value "$@"; MOSAIC="$2"; shift 2 ;;
    --mixup)
      require_value "$@"; MIXUP="$2"; shift 2 ;;
    --close-mosaic)
      require_value "$@"; CLOSE_MOSAIC="$2"; shift 2 ;;
    --cache)
      require_value "$@"; CACHE="$2"; shift 2 ;;
    --save-period)
      require_value "$@"; SAVE_PERIOD="$2"; shift 2 ;;
    --history-dirname)
      require_value "$@"; HISTORY_DIRNAME="$2"; shift 2 ;;
    --pretrain-eval-split)
      require_value "$@"; PRETRAIN_EVAL_SPLIT="$2"; shift 2 ;;
    --source)
      require_value "$@"; SOURCE="$2"; shift 2 ;;
    --preprocessed)
      require_value "$@"; PREPROCESSED="$2"; shift 2 ;;
    --data)
      require_value "$@"; DATA_YAML="$2"; PREPROCESS="0"; shift 2 ;;
    --no-preprocess)
      PREPROCESS="0"; shift ;;
    --copy-mode)
      require_value "$@"; COPY_MODE="$2"; shift 2 ;;
    --preprocess-imgsz)
      require_value "$@"; PREPROCESS_IMAGE_SIZE="$2"; shift 2 ;;
    --project)
      require_value "$@"; PROJECT="$2"; shift 2 ;;
    --name)
      require_value "$@"; NAME="$2"; shift 2 ;;
    --exist-ok)
      EXIST_OK="1"; shift ;;
    --no-save)
      SAVE_ARG=(--no-save); shift ;;
    --no-pretrain-eval)
      PRETRAIN_EVAL_ARG=(--no-pretrain-eval); shift ;;
    --no-cos-lr)
      COS_LR_ARG=(--no-cos-lr); shift ;;
    --no-amp)
      AMP_ARG=(--no-amp); shift ;;
    --no-plots)
      PLOTS_ARG=(--no-plots); shift ;;
    --no-val)
      VAL_ARG=(--no-val); shift ;;
    --no-final-val)
      FINAL_VAL_ARG=(--no-final-val); shift ;;
    --set)
      require_value "$@"; EXTRA_ARGS+=(--set "$2"); shift 2 ;;
    --install-deps)
      INSTALL_DEPS="1"; shift ;;
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

case "$SIZE" in
  n|s|m|l|x) ;;
  *)
    echo "Invalid --size '$SIZE'. Use one of: n, s, m, l, x." >&2
    exit 2 ;;
esac

case "$COPY_MODE" in
  copy|hardlink|symlink) ;;
  *)
    echo "Invalid --copy-mode '$COPY_MODE'. Use: copy, hardlink, or symlink." >&2
    exit 2 ;;
esac

if [[ -z "$DATA_YAML" ]]; then
  DATA_YAML="$PREPROCESSED/data.yaml"
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
  INSTALL_CMD=(python3 -m pip install -U -r "$ROOT_DIR/requirements.txt")
  print_command "${INSTALL_CMD[@]}"
  if [[ "$DRY_RUN" == "0" ]]; then
    "${INSTALL_CMD[@]}"
  fi
fi

if [[ "$PREPROCESS" == "1" ]]; then
  PREPROCESS_CMD=(
    python3 "$ROOT_DIR/preprocess_yolo_dataset.py"
    --source "$SOURCE"
    --output "$PREPROCESSED"
    --copy-mode "$COPY_MODE"
    --image-size "$PREPROCESS_IMAGE_SIZE"
  )
  print_command "${PREPROCESS_CMD[@]}"
  if [[ "$DRY_RUN" == "0" ]]; then
    "${PREPROCESS_CMD[@]}"
  fi
fi

TRAIN_CMD=(
  python3 "$ROOT_DIR/train_yolo26.py"
  --data "$DATA_YAML"
  --model-size "$SIZE"
  --weights-dir "$WEIGHTS_DIR"
  --epochs "$EPOCHS"
  --imgsz "$IMGSZ"
  --batch "$BATCH"
  --device "$DEVICE"
  --workers "$WORKERS"
  --patience "$PATIENCE"
  --seed "$SEED"
  --optimizer "$OPTIMIZER"
  --lr0 "$LR0"
  --lrf "$LRF"
  --momentum "$MOMENTUM"
  --weight-decay "$WEIGHT_DECAY"
  --warmup-epochs "$WARMUP_EPOCHS"
  --hsv-h "$HSV_H"
  --hsv-s "$HSV_S"
  --hsv-v "$HSV_V"
  --degrees "$DEGREES"
  --translate "$TRANSLATE"
  --scale "$SCALE"
  --fliplr "$FLIPLR"
  --mosaic "$MOSAIC"
  --mixup "$MIXUP"
  --close-mosaic "$CLOSE_MOSAIC"
  --cache "$CACHE"
  --save-period "$SAVE_PERIOD"
  --project "$PROJECT"
  --history-dirname "$HISTORY_DIRNAME"
  --pretrain-eval-split "$PRETRAIN_EVAL_SPLIT"
  "${SAVE_ARG[@]}"
  "${PRETRAIN_EVAL_ARG[@]}"
  "${COS_LR_ARG[@]}"
  "${AMP_ARG[@]}"
  "${PLOTS_ARG[@]}"
  "${VAL_ARG[@]}"
  "${FINAL_VAL_ARG[@]}"
)

if [[ -n "$NAME" ]]; then
  TRAIN_CMD+=(--name "$NAME")
fi
if [[ -n "$MODEL" ]]; then
  TRAIN_CMD+=(--model "$MODEL")
fi
if [[ -n "$RESUME" ]]; then
  TRAIN_CMD+=(--resume "$RESUME")
fi
if [[ "$EXIST_OK" == "1" ]]; then
  TRAIN_CMD+=(--exist-ok)
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

print_command "${TRAIN_CMD[@]}"
if [[ "$DRY_RUN" == "0" ]]; then
  "${TRAIN_CMD[@]}"
fi
