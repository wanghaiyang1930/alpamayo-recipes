#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# Always run from the script's own directory so relative paths
# (train_hf.py, configs/, ...) resolve regardless of caller cwd.
cd "$(dirname "$(readlink -f "$0")")"

# Default configuration
CONFIG_NAME=${CONFIG_NAME:-"sft_stage1_zero_vqa"}
NUM_GPUS=${NUM_GPUS:-4}
DATA_ROOT=${DATA_ROOT:-""}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"output_stage0_vqa"}
VLM_PATH=${VLM_PATH:-""}

usage() {
    cat <<EOF
Usage: $0 --data-root /path/to/data [options]

Train Stage1 VQA via torchrun.

Required arguments:
  --data-root PATH          Path to dataset root

Optional arguments:
  --checkpoint-path PATH    Stage1 VLM checkpoint path (default: train from pretrained)
  --vlm-path PATH           Override base VLM weights (model.vlm_name_or_path),
                            e.g. local Qwen3-VL-8B path or HF repo id
  --config NAME             Hydra config name (default: $CONFIG_NAME)
  --num-gpus N              Number of GPUs for torchrun (default: $NUM_GPUS)
  --output-dir PATH         Output directory (default: $OUTPUT_DIR)
  -h, --help                Show this help message and exit

Environment overrides:
  CONFIG_NAME, NUM_GPUS, DATA_ROOT, CHECKPOINT_PATH, OUTPUT_DIR, VLM_PATH
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --checkpoint-path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --vlm-path)
            VLM_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATA_ROOT" ]; then
    echo "Error: --data-root is required"
    usage
    exit 1
fi

echo "=========================================="
echo "Training Configuration:"
echo "  Config: $CONFIG_NAME"
echo "  GPUs: $NUM_GPUS"
echo "  Data Root: $DATA_ROOT"
echo "  Checkpoint Path: ${CHECKPOINT_PATH:-"(none, training from pretrained)"}"
echo "  Output Dir: $OUTPUT_DIR"
echo "=========================================="

# Build optional overrides
OVERRIDES=(
    "data.train_dataset.data_root=$DATA_ROOT"
    "data.val_dataset.data_root=$DATA_ROOT"
    "paths.output_dir=$OUTPUT_DIR"
    "trainer.output_dir=$OUTPUT_DIR"
)

if [ -n "$CHECKPOINT_PATH" ]; then
    OVERRIDES+=("model.stage1_vlm_checkpoint_path=$CHECKPOINT_PATH")
fi

# Run training with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_hf.py \
    --config-path configs \
    --config-name $CONFIG_NAME \
    "${OVERRIDES[@]}"
