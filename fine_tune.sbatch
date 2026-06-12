#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH -p h200p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=36G
#SBATCH --time=10:00:00
#SBATCH -J sanwa_ft
#SBATCH -o ~/logs/%x-%j.out

set -euo pipefail

# ---- User knobs (change per run) ----
CAVITY="${CAVITY:-1}"                          # 1-8
RUN_NAME="${RUN_NAME:-cavity_${CAVITY}_balanced_transformer_mlphead}"
MODEL_ID="${MODEL_ID:-cavity_${CAVITY}_balanced_ft}"

# --- Initialize Environment Modules ---
if [ -f /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
  . /usr/share/Modules/init/bash
fi

# --- Load apptainer/singularity if available ---
if command -v module >/dev/null 2>&1; then
  module avail 2>&1 | grep -qi apptainer   && module load apptainer   || true
  module avail 2>&1 | grep -qi singularity && module load singularity || true
fi
APPTAINER_BIN="$(command -v apptainer || true)"
[ -z "$APPTAINER_BIN" ] && APPTAINER_BIN="$(command -v singularity || true)"
[ -z "$APPTAINER_BIN" ] && { echo "FATAL: apptainer/singularity not found"; exit 127; }

# ---- Paths (same cluster layout as previous sbatch) ----
IMAGE="/scratch/prj0000000262/images/pytorch-2.4-cuda12.1.sif"
[ -f "$IMAGE" ] || { echo "FATAL: image not found: $IMAGE"; exit 2; }

export APPTAINER_CACHEDIR="/scratch/prj0000000262/.apptainer_cache"
mkdir -p "$APPTAINER_CACHEDIR" "$HOME/logs"

# Repo on cluster (bound to /workspace inside container)
CODE_DIR="${CODE_DIR:-$HOME/sanwa_regression_4_jun_2026_cm}"
RUNS_DIR="${RUNS_DIR:-/scratch/prj0000000262/ngyx/sanwa/runs}"
mkdir -p "$RUNS_DIR"

SCRIPT="fine_tune.py"

# Paths inside container (/workspace = CODE_DIR)
DATA_PATH="/workspace/tabular_dataset"
DATA_ID="trolley_regression_splits_combined/by_cavity/cavity_${CAVITY}/balanced"
PRETRAINED_DIR="/workspace/pretrained_models/Tiny"

EXTRA_ARGS=(
  --task_type RUL
  --data_path "$DATA_PATH"
  --data_id "$DATA_ID"
  --data_percentage "1"
  --model_id "$MODEL_ID"

  --model_arch transformer
  --regression_head mlp
  --model_type tiny
  --patch_size 64
  --dropout 0.3
  --gpu_id 0
  --use_moe false

  --load_from_pretrained true
  --pretrained_model_dir "$PRETRAINED_DIR"
  --pretraining_epoch_id 1
  --pretrained_model_type normal

  --rul_target_mode multi
  --regression_loss mae

  --num_epochs 600
  --patience 50
  --batch_size 16
  --lr 3e-4
  --wt_decay 1e-4
  --random_seed 42
)

# ---- Logging ----
echo "[INFO] Hostname: $(hostname)"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] GPUs: ${SLURM_GPUS:-N/A} | STEP_GPUS: ${SLURM_STEP_GPUS:-N/A} | GRES: ${SLURM_GRES:-N/A}"
echo "[INFO] IMAGE: $IMAGE"
echo "[INFO] CODE_DIR: $CODE_DIR"
echo "[INFO] RUNS_DIR: $RUNS_DIR"
echo "[INFO] CAVITY: $CAVITY | RUN_NAME: $RUN_NAME"
echo "[INFO] DATA_ID: $DATA_ID"
echo "[ENV ] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

# ---- Run manifest ----
cd "$CODE_DIR"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
mkdir -p "$RUN_DIR"
git rev-parse --short HEAD > "$RUN_DIR/git_commit.txt" 2>/dev/null || echo "no-git" > "$RUN_DIR/git_commit.txt"
printf "SLURM_JOB_ID=%s\nHOSTNAME=%s\nCAVITY=%s\n" "${SLURM_JOB_ID:-N/A}" "$(hostname)" "$CAVITY" > "$RUN_DIR/run_env.txt"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- Bind mounts ----
BIND_OPTS="--bind $CODE_DIR:/workspace --bind /scratch:/scratch --bind $HOME/logs:$HOME/logs"

# ---- Reusable venv & pip cache on scratch ----
VENV_DIR="/scratch/prj0000000262/ngyx/sanwa/sanwa-pl-py311"
PIP_CACHE_DIR="/scratch/prj0000000262/ngyx/sanwa/.pip_cache"
mkdir -p "$(dirname "$VENV_DIR")" "$PIP_CACHE_DIR"

EXTRA_STR=$(printf "%q " "${EXTRA_ARGS[@]}")

# ---- Execute inside container ----
srun env \
  VENV_DIR="$VENV_DIR" \
  PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  SCRIPT="$SCRIPT" \
  RUN_DIR="$RUN_DIR" \
  EXTRA_STR="$EXTRA_STR" \
  "$APPTAINER_BIN" exec --nv $BIND_OPTS "$IMAGE" bash -lc '
set -euo pipefail
cd /workspace
echo "[IN ] CWD: $(pwd)"
ls -l requirements.txt || { echo "[ERR] requirements.txt not found at /workspace"; exit 3; }
ls -l "$SCRIPT" || { echo "[ERR] script not found at /workspace/$SCRIPT"; exit 4; }

python3 -V || python -V || true

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
. "$VENV_DIR/bin/activate"

export PIP_CACHE_DIR="$PIP_CACHE_DIR"
python -m pip install --upgrade pip wheel
pip install -r requirements.txt

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "[OK ] Starting training..."
python -u "/workspace/$SCRIPT" $EXTRA_STR
'

echo "[DONE] Outputs at: $RUN_DIR"
echo "[DONE] Checkpoints also under: $CODE_DIR/checkpoints/"
