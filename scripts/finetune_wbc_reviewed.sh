#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CHECKPOINT="${PROJECT_ROOT}/runs/yolo26/txl_pbc_yolo26m2/weights/last.pt"
EPOCHS=50
IMGSZ=640
BATCH=16
LR0=0.0001
LRF=0.01
PATIENCE=15
FREEZE=10
DEVICE=""        # leave empty for auto (uses GPU if available)
PROJECT="${PROJECT_ROOT}/runs/finetune"
NAME="wbc_reviewed_ft"

# ---------------------------------------------------------------------------
# Parse optional CLI overrides
# e.g.: ./finetune_wbc_reviewed.sh --epochs 100 --batch 8 --device cpu
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --imgsz)      IMGSZ="$2";      shift 2 ;;
        --batch)      BATCH="$2";      shift 2 ;;
        --lr0)        LR0="$2";        shift 2 ;;
        --lrf)        LRF="$2";        shift 2 ;;
        --patience)   PATIENCE="$2";   shift 2 ;;
        --freeze)     FREEZE="$2";     shift 2 ;;
        --device)     DEVICE="$2";     shift 2 ;;
        --name)       NAME="$2";       shift 2 ;;
        --rebuild)    EXTRA_ARGS="${EXTRA_ARGS:-} --rebuild"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  TXL-PBC fine-tune — wbc_reviewed dataset"
echo "============================================"
echo "  Checkpoint : $CHECKPOINT"
echo "  Epochs     : $EPOCHS"
echo "  Image size : $IMGSZ"
echo "  Batch      : $BATCH"
echo "  LR0        : $LR0"
echo "  Freeze     : $FREEZE backbone layers"
echo "  Device     : ${DEVICE:-auto}"
echo "  Output     : $PROJECT/$NAME"
echo "============================================"
echo ""

cd "$PROJECT_ROOT"

DEVICE_ARG=""
[[ -n "$DEVICE" ]] && DEVICE_ARG="--device $DEVICE"

python scripts/finetune_wbc_reviewed.py \
    --checkpoint "$CHECKPOINT" \
    --epochs     "$EPOCHS"     \
    --imgsz      "$IMGSZ"      \
    --batch      "$BATCH"      \
    --lr0        "$LR0"        \
    --lrf        "$LRF"        \
    --patience   "$PATIENCE"   \
    --freeze     "$FREEZE"     \
    --project    "$PROJECT"    \
    --name       "$NAME"       \
    ${DEVICE_ARG}              \
    ${EXTRA_ARGS:-}

echo ""
echo "Done. Best checkpoint saved to:"
echo "  $PROJECT/$NAME/weights/best.pt"
