#!/bin/bash
# sanity run to test torch.compile compatibility with mHC
# should produce identical results to sanity_mhc.sh but faster

set -e

STEPS=${1:-300}
DEPTH=${2:-12}
WANDB_RUN=${WANDB_RUN:-"mhc-compile-test-$(date +%Y%m%d-%H%M%S)"}

echo "======================================"
echo "mHC Compile Sanity Run"
echo "======================================"
echo "Steps: $STEPS"
echo "Depth: $DEPTH"
echo "WandB run: $WANDB_RUN"
echo ""

cd "$(dirname "$0")"

if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
fi

echo "Starting training..."
echo ""

python -m scripts.base_train \
    --depth=$DEPTH \
    --num_iterations=$STEPS \
    --mhc_enabled=True \
    --mhc_num_streams=4 \
    --mhc_sinkhorn_iters=50 \
    --mhc_sinkhorn_tau=0.1 \
    --device_batch_size=4 \
    --total_batch_size=32768 \
    --eval_every=50 \
    --core_metric_every=-1 \
    --save_every=-1 \
    --run=$WANDB_RUN

echo ""
echo "======================================"
echo "Compile sanity run complete!"
echo "======================================"
echo ""
echo "Compare with previous non-compile run:"
echo "  - loss curve should be similar"
echo "  - mHC metrics should be similar"
echo "  - training should be ~20-30% faster"
