#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CHECKPOINT="${PROJECT_ROOT}/runs/finetune/granular_wbc_ft/weights/best.pt"
EPOCHS=60
IMGSZ=640
BATCH=16
LR0=0.0003
PATIENCE=15
FREEZE=10
DEVICE="0"
NAME="granular_wbc_ft_v2"
PROJECT="${PROJECT_ROOT}/runs/finetune"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --imgsz)      IMGSZ="$2";      shift 2 ;;
        --batch)      BATCH="$2";      shift 2 ;;
        --lr0)        LR0="$2";        shift 2 ;;
        --patience)   PATIENCE="$2";   shift 2 ;;
        --freeze)     FREEZE="$2";     shift 2 ;;
        --device)     DEVICE="$2";     shift 2 ;;
        --name)       NAME="$2";       shift 2 ;;
        --rebuild)    EXTRA_ARGS="$EXTRA_ARGS --rebuild"; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "================================================"
echo "  Granular WBC fine-tune  (round 2)"
echo "  Source : cvat_review/wbc_final_merged.zip (~9k images)"
echo "  Classes: RBC | Platelets | Neutrophil"
echo "           Lymphocyte | Monocyte | Eosinophil | Basophil"
echo "================================================"
echo "  Checkpoint : $CHECKPOINT"
echo "  Epochs     : $EPOCHS"
echo "  Batch      : $BATCH  |  LR0: $LR0"
echo "  Freeze     : $FREEZE backbone layers"
echo "  Device     : $DEVICE"
echo "  Output     : $PROJECT/$NAME"
echo "================================================"

cd "$PROJECT_ROOT"

python scripts/finetune_final_merged.py \
    --checkpoint "$CHECKPOINT" \
    --epochs     "$EPOCHS"     \
    --imgsz      "$IMGSZ"      \
    --batch      "$BATCH"      \
    --lr0        "$LR0"        \
    --patience   "$PATIENCE"   \
    --freeze     "$FREEZE"     \
    --device     "$DEVICE"     \
    --project    "$PROJECT"    \
    --name       "$NAME"       \
    $EXTRA_ARGS

echo ""
echo "Best checkpoint → ${PROJECT}/${NAME}/weights/best.pt"
