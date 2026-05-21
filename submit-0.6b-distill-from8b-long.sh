#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=12:00:00
#SBATCH --job-name=ap0.6b-distill-from8b-TOP256-long
#SBATCH --output=/capstor/store/cscs/swissai/a140/codebases/Megatron-LM-QAT/logs/slurm/training/%x-%j.out
#SBATCH --error=/capstor/store/cscs/swissai/a140/codebases/Megatron-LM-QAT/logs/slurm/training/%x-%j.err
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --partition=normal
#SBATCH --environment=/capstor/store/cscs/swissai/a140/containers/megatron.toml   # new_nemo  # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --signal=SIGUSR2@600	# Send SIGUSR2 600 seconds before hitting the time limit
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

echo "START TIME: $(date)"

################ Configs ################
# NOTE(tj.solergibert) Check the `Data` section in the README. Use `,` to specify multiple datasets e.g. "/path/to/dataset/A,/path/to/dataset/B,/path/to/dataset/C"
# Phase 5
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
echo "Using datasets: $DATASETS"

# This config trains with 504 * 4096 = 2_064_384 tokens per batch.
# Results in 512 / 2 = 256 forward passes with 4 replicas per node requires a 
# Max of 64 nodes such that each replica does batch accumulation of 1.
MBS=4 # Micro batch size
GBS=512 # Global batch size
SEQ_LEN=4096 # Sequence length 
TRAINING_STEPS=800_000  # total tokens = 160_000 * 2_097_152 = 335_544_320_000
CHECKPOINT_STEPS=20_000
LATEST_CHECKPOINT_STEPS=1_000

AUTO_JOB_REQUEUE=true # Set to `true` to continuously submit jobs to Slurm until training is complete. Enable it once you are sure of the cost involved in running this experiment.

#### Debugging ####
LOG_NCCL=false # Log NCCL_DEBUG=info. Every process will dump the logging into separate files, check `NCCL_DEBUG_FILE`
NSYS_PROFILER=false # Turn on the NSYS profiler. Check the `--profile-*` args available in megatron/training/arguments.py
MOCK_DATA=true # Set to `true` to use mock data
###################

# Megatron source and dataset cache
MEGATRON_LM_DIR=/capstor/store/cscs/swissai/a140/codebases/Megatron-LM-QAT
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache
mkdir -p $DATASET_CACHE_DIR
BACKUP_CODEBASE=false # Set to `true` to copy the codebase to the experiment folder and re-use it across runs

# Logging directories & artifacts
PROJECT_NAME=distill
EXP_NAME=ap0.6b-from8b-TOP256-long
PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME

#########################################

EXP_DIR=$PROJECT_DIR/$EXP_NAME
CKPT_DIR=$EXP_DIR/checkpoints
TRIGGER_DIR=$EXP_DIR/triggers
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
COMPUTE_ENVIRONMENT_DIR=$DEBUG_DIR/compute_environment.txt
GPU_MEM_LOGGING=$DEBUG_DIR/memory_logging.txt
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard
BACKUP_CODEBASE_DIR=$SCRATCH/backup-code/Megatron-LM # $EXP_DIR/Megatron-LM

# Set up ENV
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
export TRITON_HOME=/dev/shm/
export TRITON_CACHE_DIR=/tmp/triton

# We are preparing for torch.distributed programs so it wants:
# - MASTER_ADDR, MASTER_PORT, WORLD_SIZE - already known before `srun`
# - RANK, LOCAL_RANK - will set at `srun` command
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=7342
export WORLD_SIZE=$SLURM_NPROCS

ulimit -c 0

#### Megatron Args #### Check megatron/training/arguments.py
# Based on the Qwen3 0.6B model.
# --transformer-impl transformer_engine
# --use-precision-aware-optimizer
TRANSFORMER_ENGINE_ARGS=(
	--main-grads-dtype fp32
)

NETWORK_SIZE_ARGS=(
	--num-layers 20
	--hidden-size 1024
	--ffn-hidden-size 6144  # xielu
	--num-attention-heads 16
	--group-query-attention
	--num-query-groups 4
	--max-position-embeddings $SEQ_LEN
	--position-embedding-type rope
	--rotary-base 500000
	# --use-rope-scaling   # Dhia
	# --rope-scaling-factor 32   # Dhia
	--make-vocab-size-divisible-by 128
	--normalization RMSNorm
	--xielu  # xielu
	--qk-layernorm  # op-block
	--qknorm-impl te  # op-block
	# --untie-embeddings-and-output-weights
)

LOGGING_ARGS=(
	--log-throughput
	--log-progress
	--tensorboard-dir $TENSORBOARD_DIR
	--no-log-loss-scale-to-tensorboard
	--log-memory-to-tensorboard
)

REGULARIZATION_ARGS=(
	--attention-dropout 0.0
	--hidden-dropout 0.0
	--weight-decay 0.1
	--clip-grad 0.1  # ademamix
	--adam-beta1 0.9
	--adam-beta2 0.999  # ademamix
	--ademamix-alpha 8  # ademamix
	--ademamix-beta3 0.9999  # ademamix
	--ademamix-beta3-warmup 100000  # ademamix
	--ademamix-alpha-warmup 100000  # ademamix
)

TRAINING_ARGS=(
	--micro-batch-size $MBS
	--global-batch-size $GBS
	--no-check-for-nan-in-loss-and-grad
	--train-iters $TRAINING_STEPS
	--log-interval 1
	--cross-entropy-loss-fusion
	--disable-bias-linear
	--optimizer ademamix  # ademamix
	--dataloader-type single
	--manual-gc
	--manual-gc-interval 500
	--exit-signal-handler
	--trigger-path $TRIGGER_DIR
)

INITIALIZATION_ARGS=(
	--seed 28
	--init-method-std 0.02
)

# NOTE(tj.solergibert) Check all the arguments in megatron/training/arguments.py#L1548 or https://github.com/NVIDIA/Megatron-LM/blob/0dd78ddcdb117ce4f2e9761449274d87af717674/megatron/training/arguments.py#L1548-L1606
LEARNING_RATE_ARGS=(
	--lr 0.0006             # 6e-4
	--min-lr 0.00006       # x10 reduction
	--lr-decay-style WSD    # WSD schedule
	--lr-warmup-iters 500
	--lr-wsd-decay-style 1-sqrt  # WSD schedule
	--lr-wsd-decay-iters 160000  # ~355 BTs of cooldown
	--override-opt_param-scheduler
)

# NOTE(tj.solergibert) Check the `Checkpointing` section in the README
SAVE_CKPT_DIR=/capstor/store/cscs/swissai/infra01/distillation/checkpoints/$PROJECT_NAME/$EXP_NAME/checkpoints
mkdir -p $SAVE_CKPT_DIR
CHECKPOINTING_ARGS=(
	--save $SAVE_CKPT_DIR
	--save-interval $CHECKPOINT_STEPS
	--non-persistent-save-interval $LATEST_CHECKPOINT_STEPS
	--non-persistent-ckpt-type global
	--ckpt-format torch_dist
	--load $SAVE_CKPT_DIR
	--async-save
)

MIXED_PRECISION_ARGS=(
	--bf16
)

DISTRIBUTED_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
	--use-distributed-optimizer
	--overlap-grad-reduce
	--overlap-param-gather
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
	--num-workers 4
	--num-dataset-builder-threads 1
	--goldfish-loss  # goldfish
	--goldfish-k 50  # goldfish
	--goldfish-h 50  # goldfish
)

# Set up directories
mkdir -p $CKPT_DIR
mkdir -p $PROJECT_DIR
mkdir -p $TRIGGER_DIR
mkdir -p $DEBUG_DIR
mkdir -p $LOGGING_DIR

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
  DATA_ARGS="${DATA_ARGS[@]} --mock-data"
else
  DATA_ARGS="${DATA_ARGS[@]} --data-path $(python3 $MEGATRON_LM_DIR/scripts/tools/create_data_config.py -p $DATASETS) --data-cache-path $DATASET_CACHE_DIR" #$DATASETS
fi


CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="python3 $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    $DATA_ARGS"

# WANDB Logging
export WANDB_ENTITY=distill-quant-laws
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
cp $0 $DEBUG_DIR

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

SRUN_ARGS=" \
	-lu \
	--cpus-per-task $SLURM_CPUS_PER_TASK \
	--wait 60 \
	--jobid $SLURM_JOB_ID \
	--kill-on-bad-exit 1 \
	"

srun -lu bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\\s*[0-9]*MiB")' > $GPU_MEM_LOGGING

if [ "$AUTO_JOB_REQUEUE" = true ]; then
	echo "[$(date)] $(sbatch --dependency=singleton $0)"
fi

srun --cpus-per-task $SLURM_CPUS_PER_TASK -lu bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $CMD_PREFIX $TRAINING_CMD"

echo "END TIME: $(date)"

if [ -f $TRIGGER_DIR/exit ]; then
   echo "[$(date)] Detected exit trigger in $TRIGGER_DIR/exit, cancelling pending jobs"
   rm -rf $TRIGGER_DIR/exit  
   scancel --jobname $SLURM_JOB_NAME
fi