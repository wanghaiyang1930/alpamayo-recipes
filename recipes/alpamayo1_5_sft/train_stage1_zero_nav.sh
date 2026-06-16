#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Usage:
#    cd /home/workspace/source/alpamayo-recipes && source ./recipes/alpamayo1_5_sft/.venv/bin/activate && bash ./recipes/alpamayo1_5_sft/train_stage1_zero_nav.sh  --num-gpus 2 --local-dir /home/workspace/source/data/nvidia/PhysicalAI-Autonomous-Vehicles --annotations-path /home/workspace/source/data/nvidia/PhysicalAI-Autonomous-Vehicles/nav/nav_samples.json --vlm-path /home/workspace/source/data/Qwen/Qwen3-VL-8B-Instruct
set -e

# Always run from the script's own directory so relative paths
# (train_hf.py, configs/, ...) resolve regardless of caller cwd.
cd "$(dirname "$(readlink -f "$0")")"

# Default configuration
CONFIG_NAME=${CONFIG_NAME:-"sft_stage1_zero_nav"}
NUM_GPUS=${NUM_GPUS:-4}
LOCAL_DIR=${LOCAL_DIR:-""}
ANNOTATIONS_PATH=${ANNOTATIONS_PATH:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"output_stage1_zero_nav"}
VLM_PATH=${VLM_PATH:-""}

usage() {
    cat <<EOF
Usage: $0 --local-dir /path/to/pai --annotations-path /path/to/nav_samples.json [options]

Train Stage1 Nav from scratch via torchrun.

Required arguments:
  --local-dir PATH          Path to PAI dataset
  --annotations-path PATH   Path to nav annotations JSON

Optional arguments:
  --vlm-path PATH           Override base VLM weights (model.vlm_name_or_path),
                            e.g. local Qwen3-VL-8B path or HF repo id
  --config NAME             Hydra config name (default: $CONFIG_NAME)
  --num-gpus N              Number of GPUs for torchrun (default: $NUM_GPUS)
  --output-dir PATH         Output directory (default: $OUTPUT_DIR)
  -h, --help                Show this help message and exit

Environment overrides:
  CONFIG_NAME, NUM_GPUS, LOCAL_DIR, ANNOTATIONS_PATH, OUTPUT_DIR, VLM_PATH
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
        --local-dir)
            LOCAL_DIR="$2"
            shift 2
            ;;
        --annotations-path)
            ANNOTATIONS_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --vlm-path)
            VLM_PATH="$2"
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
if [ -z "$LOCAL_DIR" ]; then
    echo "Error: --local-dir is required (path to PAI dataset)"
    usage
    exit 1
fi

if [ -z "$ANNOTATIONS_PATH" ]; then
    echo "Error: --annotations-path is required (path to nav annotations JSON)"
    usage
    exit 1
fi

echo "=========================================="
echo "Training Configuration (Nav from scratch):"
echo "  Config: $CONFIG_NAME"
echo "  GPUs: $NUM_GPUS"
echo "  Local Dir: $LOCAL_DIR"
echo "  Annotations: $ANNOTATIONS_PATH"
echo "  Output Dir: $OUTPUT_DIR"
echo "  VLM Path: ${VLM_PATH:-"(default from config)"}"
echo "=========================================="

# Build overrides
OVERRIDES=(
    "data.train_dataset.local_dir=$LOCAL_DIR"
    "data.train_dataset.annotations_path=$ANNOTATIONS_PATH"
    "data.val_dataset.local_dir=$LOCAL_DIR"
    "data.val_dataset.annotations_path=$ANNOTATIONS_PATH"
    "paths.output_dir=$OUTPUT_DIR"
    "trainer.output_dir=$OUTPUT_DIR"
)

if [ -n "$VLM_PATH" ]; then
    OVERRIDES+=("model.vlm_name_or_path=$VLM_PATH")
fi

# Run training with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_hf.py \
    --config-path configs \
    --config-name $CONFIG_NAME \
    "${OVERRIDES[@]}"
