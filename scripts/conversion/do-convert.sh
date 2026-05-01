#!/bin/bash

# runs the following checkpoint conversions: 
#   - torch_dist           ---> torch ,  if CKPT_IS_TORCH_DIST=true.
#   - core (torch backend) ---> HF    ,  always.

# pip install git+https://github.com/swiss-ai/transformers.git@model/swissai#egg=transformers
pip install transformers==4.57.6

MEGATRON_LM_DIR=/capstor/store/cscs/swissai/a140/codebases/Megatron-LM-QAT
CKPT_STEP=794000
CKPT_PATH=/capstor/store/cscs/swissai/infra01/distillation/checkpoints/distill/ap0.6b-from8b-TOP256-foravg/checkpoints

# [torch_dist -> torch] dependencies
CKPT_IS_TORCH_DIST=true
TORCH_DIST_SCRIPT=$MEGATRON_LM_DIR/scripts/conversion/torchdist_2_torch.py
TORCH_CKPT_SAVE_PATH=/iopsstor/scratch/cscs/$USER/checkpoints
# [core (torch) --> HF] dependencies
HF_SAVE_DIR=/capstor/store/cscs/swissai/a140/checkpoints/huggingface
SAVE_DIR=$HF_SAVE_DIR/ap0.6b-avg/ap0.6b-avg-it0$CKPT_STEP
mkdir -p $HF_SAVE_DIR
LOADER=core
SAVER=swissai_hf


# Run torch_dist --> torch
if [[ "$CKPT_IS_TORCH_DIST" == true ]]; then
    LOAD_DIR=$TORCH_CKPT_SAVE_PATH/torch
    echo "Running torch_dist --> torch conversion..."
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun $TORCH_DIST_SCRIPT \
    --bf16 \
    --load $CKPT_PATH \
    --ckpt-step $CKPT_STEP \
    --ckpt-convert-save $TORCH_CKPT_SAVE_PATH
else
    LOAD_DIR=$CKPT_PATH
    echo "Skipping torch_dist --> torch conversion..."
fi


# Run core --> HF
echo "Running core --> HF conversion..."
python $MEGATRON_LM_DIR/tools/checkpoint/convert.py \
    --model-type GPT \
    --loader  $LOADER \
    --saver $SAVER \
    --load-dir $LOAD_DIR \
    --save-dir $SAVE_DIR \
    --hf-tokenizer alehc/swissai-tokenizer
