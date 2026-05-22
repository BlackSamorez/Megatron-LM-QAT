#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=12:00:00
#SBATCH --job-name=ap8b-logits
#SBATCH --output=/capstor/store/cscs/swissai/a140/codebases/Megatron-LM-Logits/logs/slurm/logits/%x-%j.out
#SBATCH --error=/capstor/store/cscs/swissai/a140/codebases/Megatron-LM-Logits/logs/slurm/logits/%x-%j.err
#SBATCH --partition=normal
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/capstor/store/cscs/swissai/a140/containers/megatron.toml
#SBATCH --signal=SIGUSR2@600	# Send SIGUSR2 600 seconds before hitting the time limit
#SBATCH --no-requeue			# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

echo "START TIME: $(date)"

################ Configs ################
# NOTE(tj.solergibert) Check the `Data` section in the README. Use `,` to specify multiple datasets e.g. "/path/to/dataset/A,/path/to/dataset/B,/path/to/dataset/C"
# Phase 5 only | Approx 1.77T tokens | Cooldown, remember to modify the LR decay & CO
DATAROOT=/iopsstor/scratch/cscs/jpcoles/a06
DATASETS=(
      $DATAROOT/phase-5/finemath-3plus-merge
      $DATAROOT/phase-5/infiwebmath-3plus-fine-merge
      $DATAROOT/phase-5/starcoder-extras-merge
      $DATAROOT/phase-5/starcoder-threshold-0-merge
      $DATAROOT/phase-5/swissai-dclm-edu-filterrobots_fine-merge
      $DATAROOT/phase-5/swissai-fineweb-2-quality_10-filterrobots-merge
      $DATAROOT/phase-5/swissai-megamath-web-pro-filterrobots-merge
      $DATAROOT/phase-5/clean-wikipedia
      $DATAROOT/phase-5/parallel-v2
      $DATAROOT/phase-5/triplicate/provenance-flan-single-replica-1
      $DATAROOT/phase-5/triplicate/euroblocks-templated-1
      $DATAROOT/phase-5/triplicate/provenance-flan-single-replica-2
      $DATAROOT/phase-5/triplicate/euroblocks-templated-2
      $DATAROOT/phase-5/triplicate/provenance-flan-single-replica-3
      $DATAROOT/phase-5/triplicate/euroblocks-templated-3
      $DATAROOT/phase-5/roman/merged/stackv1/threshold_2
      $DATAROOT/phase-5/roman/merged/stackv1/threshold_3
      $DATAROOT/phase-5/roman/merged/stackv2/threshold_0
)
DATASETS=$(IFS=','; echo "${DATASETS[*]}")

MBS=32 			# Micro batch size
TP=1 			# Tensor parallelism
SEQ_LEN=4096 	# Sequence length
LOGITS_SAMPLES=429998080  # Number of sequences to produce logits for
LOGITS_SAMPLES_PER_CHUNK=32  # Number if sequences per file
LOGITS_TOPK=256           # Top-K logits to save
LOGITS_SAVE_DIR=/capstor/scratch/cscs/blacksamorez/logits/8B_TOP${LOGITS_TOPK}_logits_gzip
# LOGITS_SAVE_DIR=/capstor/store/cscs/swissai/infra01/distillation/8B_TOP${LOGITS_TOPK}_logits_gzip
LOAD_DIR=/capstor/scratch/cscs/asolergi/main_run_70B_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-70b-512-nodes-1e-5lr/8b-checkpoints-v4

export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1024

AUTO_JOB_REQUEUE=false 

#### Debugging ####
LOG_NCCL=false
NSYS_PROFILER=false
MOCK_DATA=false # <- HERE
###################

# Megatron source and dataset cache
MEGATRON_LM_DIR=/capstor/store/cscs/swissai/a140/codebases/Megatron-LM-Logits
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/blacksamorez/datasets/cache-new
BACKUP_CODEBASE=false

# Logging directories & artifacts (debug paths under /blacksamorez)
PROJECT_NAME=Apertus-logits
EXP_NAME=8B-$SLURM_NNODES-nodes-phase5-mix
PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME

#########################################

EXP_DIR=$PROJECT_DIR/$EXP_NAME
TORCH_INDUCTOR_CACHE_DIR=/workspace/torch_compile_cache/$SLURM_JOB_ID
TRITON_HOME_CACHE_DIR=/workspace/triton_home_cache/$SLURM_JOB_ID
PYTHON_CACHE_DIR=/workspace/python_cache/$SLURM_JOB_ID
TRIGGER_DIR=$EXP_DIR/triggers
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
COMPUTE_ENVIRONMENT_DIR=$DEBUG_DIR/compute_environment.txt
GPU_MEM_LOGGING=$DEBUG_DIR/memory_logging.txt
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard
BACKUP_CODEBASE_DIR=$EXP_DIR/Megatron-LM

# Set up ENV
export WANDB__FILE_STREAM_RETRY_MAX=10
export HF_HUB_OFFLINE=1
export WANDB_ENTITY=distill-quant-laws
export WANDB_PROJECT="${PROJECT_NAME}"

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# We are preparing for torch.distributed programs so it wants:
# - MASTER_ADDR, MASTER_PORT, WORLD_SIZE - already known before `srun`
# - RANK, LOCAL_RANK - will set at `srun` command
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8888
export WORLD_SIZE=$SLURM_NPROCS

ulimit -c 0

#### Megatron Args #### Check megatron/training/arguments.py
# Based on the Llama 3.2 8B model.
# --transformer-impl transformer_engine
# --use-precision-aware-optimizer
TRANSFORMER_ENGINE_ARGS=(
	--main-grads-dtype fp32
)

NETWORK_SIZE_ARGS=(
	--num-layers 32
	--hidden-size 4096
	--ffn-hidden-size 21504  # xielu
	--num-attention-heads 32
	--group-query-attention
	--num-query-groups 8
	--max-position-embeddings $SEQ_LEN
	--position-embedding-type rope
	--rotary-base 500000
	--use-rope-scaling
	--rope-scaling-factor 8
	--make-vocab-size-divisible-by 128
	--normalization RMSNorm
	--xielu  # xielu
	--qk-layernorm  # op-block
	--qknorm-impl apex  # op-block
	--untie-embeddings-and-output-weights
	# --fix-old-xielu
)

LOGGING_ARGS=(
	--log-throughput
	--log-progress
	--tensorboard-dir $TENSORBOARD_DIR
	--no-log-loss-scale-to-tensorboard
	--log-memory-to-tensorboard
	--log-params-norm
)

REGULARIZATION_ARGS=(
	--attention-dropout 0.0
	--hidden-dropout 0.0
)

LOGITS_ARGS=(
	--micro-batch-size $MBS
	--logits_samples $LOGITS_SAMPLES
    --logits_top_k $LOGITS_TOPK
    --logits_samples_per_chunk $LOGITS_SAMPLES_PER_CHUNK
    --logits_save_dir $LOGITS_SAVE_DIR
	--log-interval 1
	--disable-bias-linear
	--dataloader-type single
	--manual-gc
	--manual-gc-interval 500
	--exit-signal-handler
	--trigger-path $TRIGGER_DIR
	--distributed-timeout-minutes 10
)

INITIALIZATION_ARGS=(
	--seed 28
	--init-method-std 0.008944
	--no-load-rng
)

# NOTE(tj.solergibert) Check the `Checkpointing` section in the README
CHECKPOINTING_ARGS=(
	--ckpt-format torch_dist
	--load $LOAD_DIR
	--async-save
)

MIXED_PRECISION_ARGS=(
	--bf16
)

DISTRIBUTED_ARGS=(
	--tensor-model-parallel-size $TP
	--pipeline-model-parallel-size 1
)

TOKENIZER_ARGS=(
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model alehc/swissai-tokenizer
)

DATA_ARGS=(
	--split 100,0,0
	--seq-length $SEQ_LEN
	--reset-position-ids  # crossDocAttn
	--reset-attention-mask  # crossDocAttn
	--eod-mask-loss  # crossDocAttn
	--num-workers 32
	--num-dataset-builder-threads 1
)

# Set up directories
mkdir -p $LOAD_DIR
mkdir -p $PROJECT_DIR
mkdir -p $TRIGGER_DIR
mkdir -p $DEBUG_DIR
mkdir -p $LOGGING_DIR
mkdir -p $TORCH_INDUCTOR_CACHE_DIR
mkdir -p $TRITON_HOME_CACHE_DIR
mkdir -p $PYTHON_CACHE_DIR

# Adding Exit trigger detection before the job JIC we aren't able to finish the first iteration
if [ -f $TRIGGER_DIR/exit ]; then
   echo "[$(date)] Detected exit trigger in $TRIGGER_DIR/exit, cancelling pending jobs"
   rm -rf $TRIGGER_DIR/exit  
   scancel --jobname $SLURM_JOB_NAME
fi

# Backup codebase
if [ "$BACKUP_CODEBASE" == true ]; then
  if [ -z "$(ls -A "$BACKUP_CODEBASE_DIR")" ]; then
  	echo "[$(date)] Copying codebase in $MEGATRON_LM_DIR to $BACKUP_CODEBASE_DIR..."
  	rsync -av --exclude-from=$MEGATRON_LM_DIR/.gitignore $MEGATRON_LM_DIR/ $BACKUP_CODEBASE_DIR/ &> /dev/null
  fi
  MEGATRON_LM_DIR=$BACKUP_CODEBASE_DIR
fi

echo "[$(date)] Using codebase in $MEGATRON_LM_DIR"

cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

# Data Args
if [ "$MOCK_DATA" = true ]; then
  DATA_ARGS="${DATA_ARGS[@]} --mock-data --data-cache-path $DATASET_CACHE_DIR-mock"
else
  DATA_ARGS="${DATA_ARGS[@]} --data-path $(python3 $MEGATRON_LM_DIR/scripts/tools/create_data_config.py -p $DATASETS) --data-cache-path $DATASET_CACHE_DIR"
fi

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="python3 $MEGATRON_LM_DIR/get_logits.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${LOGITS_ARGS[@]} \
    $DATA_ARGS"

# WANDB Logging
if [ -n "$WANDB_API_KEY" ]; then
  echo "[$(date)] WANDB API key detected. Enabling WANDB logging."
  # Sync any previous run data if present
  if [ -d "$LOGGING_DIR/wandb/latest-run" ]; then
    echo "[$(date)] Syncing WANDB from previous run"
    wandb sync "$LOGGING_DIR/wandb/latest-run"
  fi
  # Add wandb-related args to TRAINING_CMD
  TRAINING_CMD="$TRAINING_CMD \
    --wandb-save-dir $LOGGING_DIR \
    --wandb-project $PROJECT_NAME \
    --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
  export WANDB_MODE=disabled
  echo "[$(date)] No WANDB API key found. WANDB logging disabled."
fi

# NCCL Debug
if [ "$LOG_NCCL" = true ]; then
  CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-hostname-\$SLURMD_NODENAME-local-rank-\$SLURM_LOCALID-procid-\$SLURM_PROCID.txt $CMD_PREFIX"
fi

# NSYS profiler
if [ "$NSYS_PROFILER" = true ]; then
    NSYS_LAUNCHER="nsys profile -s none --trace='nvtx,cudnn,cublas,cuda' --output=$DEBUG_DIR/nsys-trace-hostname-\$SLURMD_NODENAME-procid-\$SLURM_PROCID.nsys-rep --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
    TRAINING_CMD="$NSYS_LAUNCHER $TRAINING_CMD --profile"
fi

# Save sbatch script
cp $0 $DEBUG_DIR/slurm-script.sh
chmod 777 $DEBUG_DIR/slurm-script.sh

# Clean triggers
rm -f $TRIGGER_DIR/save
rm -f $TRIGGER_DIR/exit

# Checkpoint Compute Environment
echo -e "$(date)" > $COMPUTE_ENVIRONMENT_DIR 
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nCMD: $CMD_PREFIX $TRAINING_CMD" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nSlurm file: $0\n" >> $COMPUTE_ENVIRONMENT_DIR
cat $0 >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nTOML file: $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment\n" >> $COMPUTE_ENVIRONMENT_DIR
cat $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nNODES: $(scontrol show hostnames $SLURM_JOB_NODELIST)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nMegatron path: $MEGATRON_LM_DIR ($(git -C $MEGATRON_LM_DIR rev-parse --verify HEAD))" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\n$(pip list)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\n$(nvidia-smi)" >> $COMPUTE_ENVIRONMENT_DIR # CUDA Version & Driver
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nEnvironment Variables:\n\n$(printenv)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 

srun -lu bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\\s*[0-9]*MiB")' > $GPU_MEM_LOGGING

if [ "$AUTO_JOB_REQUEUE" = true ]; then
	echo "[$(date)] $(sbatch --dependency=singleton $0)"
fi

srun --cpus-per-task $SLURM_CPUS_PER_TASK \
	-lu bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID TORCHINDUCTOR_CACHE_DIR=$TORCH_INDUCTOR_CACHE_DIR/cache_\$SLURM_PROCID TRITON_HOME=$TRITON_HOME_CACHE_DIR/cache_\$SLURM_PROCID PYTHONPYCACHEPREFIX=$PYTHON_CACHE_DIR/cache_\$SLURM_PROCID $CMD_PREFIX $TRAINING_CMD"

# Remove Torchinductor, Triton & Python caches
rm -rf $TORCH_INDUCTOR_CACHE_DIR
rm -rf $TRITON_HOME_CACHE_DIR
rm -rf $PYTHON_CACHE_DIR

echo "END TIME: $(date)"

if [ -f $TRIGGER_DIR/exit ]; then
   echo "[$(date)] Detected exit trigger in $TRIGGER_DIR/exit, cancelling pending jobs"
   rm -rf $TRIGGER_DIR/exit  
   scancel --jobname $SLURM_JOB_NAME
fi
