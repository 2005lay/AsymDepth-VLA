# AsymDepth-VLA

AsymDepth-VLA is a research codebase for lightweight Vision-Language-Action (VLA) models. Built on VLA-Adapter, it constructs compact student models through asymmetric depth pruning and teacher knowledge distillation, with training and evaluation support for LIBERO and CALVIN.

## Highlights

- Asymmetric pruning: the VLM backbone is reduced from 24 to 16 layers, while the Policy network is reduced from 24 to 8 layers.
- Joint distillation: both action-generation distillation and intermediate-representation distillation are used.
- Complete checkpoints: the training script merges LoRA weights and saves directly evaluable Hugging Face checkpoints.
- Unified evaluation: success rate, VLM latency, Policy latency, total latency, and peak GPU memory are reported.

## Pruning Configuration

```text
VLM layers:    1,2,3,5,7,9,10,11,13,15,17,19,21,22,23,24
Policy layers: 1,4,7,10,13,16,20,24
```

The default distillation objective is:

```text
L = L_task + 1.0 * L_action_KD + 0.5 * L_rep_KD
```

The commands below use `teacher_sliced`: the student inherits the selected teacher layers and then recovers performance through the task objective and teacher distillation. This is the complete pruned method rather than the “without teacher-weight initialization” ablation.

## Repository Structure

```text
AsymDepth-VLA/
├── prismatic/                 # Core model, data-processing, and training code
├── vla-scripts/               # Fine-tuning, CALVIN evaluation, and weight conversion
├── experiments/robot/libero/  # LIBERO closed-loop evaluation
├── scripts/                   # Python pretraining and data-processing utilities
├── tools/                     # Pruning-logic and checkpoint validation tools
├── pretrained_models/configs/ # Model, processor, and tokenizer configurations
├── pyproject.toml
└── LICENSE
```

## Installation

Linux, Python 3.10, a CUDA-capable GPU, and a PyTorch environment with BF16 support are recommended.

```bash
conda create -n asymdepth-vla python=3.10 -y
conda activate asymdepth-vla

cd /path/to/AsymDepth-VLA
pip install -e .
pip install "flash-attn==2.5.5" --no-build-isolation
```

If the server uses a different CUDA/PyTorch combination, install the matching FlashAttention wheel. See `pyproject.toml` for the complete Python dependency list.

Set the shared paths before downloading data or running experiments:

```bash
export PROJECT_ROOT=/path/to/AsymDepth-VLA
export DATA_ROOT=$PROJECT_ROOT/data
export ASSET_ROOT=$PROJECT_ROOT/pretrained_models
export LIBERO_ROOT=$PROJECT_ROOT/third_party/LIBERO
export CALVIN_ROOT=$PROJECT_ROOT/third_party/calvin

export PYTHONPATH=$PROJECT_ROOT:$LIBERO_ROOT:$CALVIN_ROOT:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

mkdir -p "$DATA_ROOT" "$ASSET_ROOT" "$PROJECT_ROOT/outputs" "$PROJECT_ROOT/eval_logs"
cd "$PROJECT_ROOT"
```

## Dataset Download

### 1. LIBERO Environment and Four RLDS Suites

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO):

```bash
mkdir -p "$PROJECT_ROOT/third_party"
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_ROOT"
pip install -e "$LIBERO_ROOT"
pip install -r "$PROJECT_ROOT/experiments/robot/libero/libero_requirements.txt"
```

Download the [modified LIBERO RLDS datasets from OpenVLA](https://huggingface.co/datasets/openvla/modified_libero_rlds). The repository is approximately 10.2 GB and contains the Spatial, Object, Goal, and Long suites:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openvla/modified_libero_rlds",
    repo_type="dataset",
    local_dir="data/libero",
    max_workers=4,
)
PY
```

The downloaded directory should have the following layout:

```text
data/libero/
├── libero_spatial_no_noops/1.0.0/
├── libero_object_no_noops/1.0.0/
├── libero_goal_no_noops/1.0.0/
└── libero_10_no_noops/1.0.0/
```

### 2. CALVIN Environment, Evaluation Data, and RLDS Training Data

Install [CALVIN](https://github.com/mees/calvin):

```bash
mkdir -p "$PROJECT_ROOT/third_party"
git clone --recurse-submodules https://github.com/mees/calvin.git "$CALVIN_ROOT"
cd "$CALVIN_ROOT"

# If pyhash installation fails, first run: pip install setuptools==57.5.0
sh install.sh
```

Download the original CALVIN data required for ABC→D closed-loop evaluation:

```bash
cd "$CALVIN_ROOT/dataset"
sh download_data.sh ABC

test -d "$CALVIN_ROOT/dataset/task_ABC_D/validation"
test -f "$CALVIN_ROOT/dataset/task_ABC_D/validation/.hydra/merged_config.yaml"
```

Training uses the [CALVIN ABC RLDS dataset](https://huggingface.co/datasets/zhouhongyi/calvin_abc_rlds), which is approximately 58 GB. The code and released teacher checkpoint consistently use the key `calvin_abc_rlds`:

```bash
cd "$PROJECT_ROOT"
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="zhouhongyi/calvin_abc_rlds",
    repo_type="dataset",
    local_dir="data/calvin_abc_rlds/1.0.0",
    max_workers=4,
)
PY
```

## Base VLM and Teacher Checkpoints

Training requires the generic Prismatic VLM and a full 24V/24P Pro teacher for each benchmark suite. The following command downloads the base VLM and all five teacher checkpoints:

```bash
cd "$PROJECT_ROOT"
python - <<'PY'
from huggingface_hub import snapshot_download

models = {
    "Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b":
        "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b",
    "VLA-Adapter/LIBERO-Spatial-Pro": "pretrained_models/LIBERO-Spatial-Pro",
    "VLA-Adapter/LIBERO-Object-Pro":  "pretrained_models/LIBERO-Object-Pro",
    "VLA-Adapter/LIBERO-Goal-Pro":    "pretrained_models/LIBERO-Goal-Pro",
    "VLA-Adapter/LIBERO-Long-Pro":    "pretrained_models/LIBERO-Long-Pro",
    "VLA-Adapter/CALVIN-ABC-Pro":     "pretrained_models/CALVIN-ABC-Pro",
}

for repo_id, local_dir in models.items():
    print(f"\n===== Downloading {repo_id} =====")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, max_workers=4)
PY
```

Teacher-checkpoint references:

- [LIBERO-Spatial-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Spatial-Pro)
- [LIBERO-Object-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Object-Pro)
- [LIBERO-Goal-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Goal-Pro)
- [LIBERO-Long-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Long-Pro)
- [CALVIN-ABC-Pro](https://huggingface.co/VLA-Adapter/CALVIN-ABC-Pro)

Before the first training run, cache the DINOv2 and SigLIP vision weights used by the code and validate the pruning implementation:

```bash
cd "$PROJECT_ROOT"
python tools/cache_generic_vision_backbones.py
python tools/check_pruning_logic.py
```

## Training the Pruned Model on the Four LIBERO Suites

The suite-to-dataset and teacher mapping is:

| Suite | Training dataset | Teacher directory | Evaluation suite |
|---|---|---|---|
| Spatial | `libero_spatial_no_noops` | `LIBERO-Spatial-Pro` | `libero_spatial` |
| Object | `libero_object_no_noops` | `LIBERO-Object-Pro` | `libero_object` |
| Goal | `libero_goal_no_noops` | `LIBERO-Goal-Pro` | `libero_goal` |
| Long | `libero_10_no_noops` | `LIBERO-Long-Pro` | `libero_10` |

The function below trains for 5,000 optimizer steps on one GPU with a learning rate of `2e-5` and an effective batch size of `4 × 4 = 16`:

```bash
cd "$PROJECT_ROOT"

VLM_LAYERS="1,2,3,5,7,9,10,11,13,15,17,19,21,22,23,24"
POLICY_LAYERS="1,4,7,10,13,16,20,24"

train_libero () {
  LABEL="$1"
  DATASET="$2"
  TEACHER="$3"
  RUN_ID="LIBERO_${LABEL}_AsymDepth_16V8P_5k"

  python -u tools/check_checkpoint_runtime.py \
    --checkpoint "$ASSET_ROOT/$TEACHER" \
    --teacher \
    --training-config "$ASSET_ROOT/configs" \
    --unnorm-key "$DATASET"

  python -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=1 \
    vla-scripts/finetune.py \
    --vlm_path "$ASSET_ROOT/prism-qwen25-extra-dinosiglip-224px-0_5b" \
    --config_file_path "$ASSET_ROOT/configs" \
    --data_root_dir "$DATA_ROOT/libero" \
    --dataset_name "$DATASET" \
    --run_root_dir "$PROJECT_ROOT/outputs" \
    --run_id_override "$RUN_ID" \
    --use_minivlm True \
    --use_pro_version True \
    --use_l1_regression True \
    --use_spline_head False \
    --use_film False \
    --use_fz False \
    --num_images_in_input 2 \
    --use_proprio True \
    --image_aug True \
    --vlm_kept_layers "$VLM_LAYERS" \
    --policy_kept_layers "$POLICY_LAYERS" \
    --student_init_mode teacher_sliced \
    --inherit_teacher_weights True \
    --use_distill True \
    --teacher_ckpt "$ASSET_ROOT/$TEACHER" \
    --lambda_action 1.0 \
    --lambda_rep 0.5 \
    --enable_gate_pruning False \
    --use_lora True \
    --lora_rank 64 \
    --merge_lora_during_training True \
    --batch_size 4 \
    --grad_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_warmup_steps 0 \
    --num_steps_before_decay 20000 \
    --max_steps 5000 \
    --save_freq 5000 \
    --save_steps "5000" \
    --save_latest_checkpoint_only False \
    --seed 7 \
    --check_gradients True \
    --verify_checkpoint True \
    --phase Training
}
```

Train the four suites sequentially:

```bash
mkdir -p "$PROJECT_ROOT/eval_logs/train"

train_libero Spatial libero_spatial_no_noops LIBERO-Spatial-Pro \
  2>&1 | tee "$PROJECT_ROOT/eval_logs/train/libero_spatial.log"

train_libero Object libero_object_no_noops LIBERO-Object-Pro \
  2>&1 | tee "$PROJECT_ROOT/eval_logs/train/libero_object.log"

train_libero Goal libero_goal_no_noops LIBERO-Goal-Pro \
  2>&1 | tee "$PROJECT_ROOT/eval_logs/train/libero_goal.log"

train_libero Long libero_10_no_noops LIBERO-Long-Pro \
  2>&1 | tee "$PROJECT_ROOT/eval_logs/train/libero_long.log"
```

The final directly evaluable checkpoints are written to:

```text
outputs/LIBERO_Spatial_AsymDepth_16V8P_5k--5000_chkpt
outputs/LIBERO_Object_AsymDepth_16V8P_5k--5000_chkpt
outputs/LIBERO_Goal_AsymDepth_16V8P_5k--5000_chkpt
outputs/LIBERO_Long_AsymDepth_16V8P_5k--5000_chkpt
```

The training script refuses to overwrite an existing run directory. Change `RUN_ID` or move the previous experiment before rerunning it.

## LIBERO 10×10 Closed-Loop Evaluation

Each suite contains 10 tasks. The following configuration evaluates every task 10 times, for 100 episodes per suite:

```bash
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/eval_logs/libero"

eval_libero () {
  LABEL="$1"
  TASK_SUITE="$2"
  DATASET="$3"
  CKPT="$PROJECT_ROOT/outputs/LIBERO_${LABEL}_AsymDepth_16V8P_5k--5000_chkpt"
  LOG_DIR="$PROJECT_ROOT/eval_logs/libero/$LABEL"
  mkdir -p "$LOG_DIR"

  python -u tools/check_checkpoint_runtime.py \
    --checkpoint "$CKPT" \
    --training-config "$ASSET_ROOT/configs" \
    --unnorm-key "$DATASET"

  python -u experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name "$TASK_SUITE" \
    --use_minivlm True \
    --use_l1_regression True \
    --use_proprio True \
    --num_images_in_input 2 \
    --use_film False \
    --use_pro_version True \
    --center_crop True \
    --num_open_loop_steps 8 \
    --num_trials_per_task 10 \
    --initial_states_path DEFAULT \
    --local_log_dir "$LOG_DIR" \
    --require_component_timings True \
    --save_version asymdepth-vla \
    --phase Inference
}
```

Evaluate all four suites:

```bash
eval_libero Spatial libero_spatial libero_spatial_no_noops \
  2>&1 | tee "$PROJECT_ROOT/eval_logs/libero/Spatial_eval100.log"

eval_libero Object libero_object libero_object_no_noops \
  2>&1 | tee "$PROJECT_ROOT/eval_logs/libero/Object_eval100.log"

eval_libero Goal libero_goal libero_goal_no_noops \
  2>&1 | tee "$PROJECT_ROOT/eval_logs/libero/Goal_eval100.log"

eval_libero Long libero_10 libero_10_no_noops \
  2>&1 | tee "$PROJECT_ROOT/eval_logs/libero/Long_eval100.log"
```

The evaluation log and JSON metrics contain the success rate, VLM latency, Policy latency, end-to-end total latency, and peak CUDA memory.

## Training the Pruned Model on CALVIN

CALVIN uses the same 16V/8P pruning configuration. The following example trains for 15,000 steps and saves complete checkpoints at steps 5,000, 10,000, and 15,000:

```bash
cd "$PROJECT_ROOT"

VLM_LAYERS="1,2,3,5,7,9,10,11,13,15,17,19,21,22,23,24"
POLICY_LAYERS="1,4,7,10,13,16,20,24"
CALVIN_TEACHER="$ASSET_ROOT/CALVIN-ABC-Pro"
CALVIN_RUN_ID="CALVIN_ABC_AsymDepth_16V8P_15k"

python -u tools/check_checkpoint_runtime.py \
  --checkpoint "$CALVIN_TEACHER" \
  --teacher \
  --training-config "$ASSET_ROOT/configs" \
  --unnorm-key calvin_abc_rlds

python -u -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=1 \
  vla-scripts/finetune.py \
  --vlm_path "$ASSET_ROOT/prism-qwen25-extra-dinosiglip-224px-0_5b" \
  --config_file_path "$ASSET_ROOT/configs" \
  --data_root_dir "$DATA_ROOT" \
  --dataset_name calvin_abc_rlds \
  --run_root_dir "$PROJECT_ROOT/outputs" \
  --run_id_override "$CALVIN_RUN_ID" \
  --use_minivlm True \
  --use_pro_version True \
  --use_l1_regression True \
  --use_spline_head False \
  --use_film False \
  --use_fz False \
  --num_images_in_input 2 \
  --use_proprio True \
  --image_aug True \
  --vlm_kept_layers "$VLM_LAYERS" \
  --policy_kept_layers "$POLICY_LAYERS" \
  --student_init_mode teacher_sliced \
  --inherit_teacher_weights True \
  --use_distill True \
  --teacher_ckpt "$CALVIN_TEACHER" \
  --lambda_action 1.0 \
  --lambda_rep 0.5 \
  --enable_gate_pruning False \
  --use_lora True \
  --lora_rank 64 \
  --merge_lora_during_training True \
  --batch_size 4 \
  --grad_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --lr_warmup_steps 0 \
  --num_steps_before_decay 20000 \
  --max_steps 15000 \
  --save_freq 5000 \
  --save_steps "5000,10000,15000" \
  --save_latest_checkpoint_only False \
  --seed 7 \
  --check_gradients True \
  --verify_checkpoint True \
  --phase Training \
  2>&1 | tee "$PROJECT_ROOT/eval_logs/CALVIN_ABC_AsymDepth_train.log"
```

The final checkpoint is written to:

```text
outputs/CALVIN_ABC_AsymDepth_16V8P_15k--15000_chkpt
```

## CALVIN ABC→D Closed-Loop Evaluation

The official protocol evaluates 1,000 five-task sequences:

```bash
cd "$PROJECT_ROOT"
export CALVIN_ROOT=$PROJECT_ROOT/third_party/calvin
export PYTHONPATH=$PROJECT_ROOT:$CALVIN_ROOT:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

CALVIN_CKPT="$PROJECT_ROOT/outputs/CALVIN_ABC_AsymDepth_16V8P_15k--15000_chkpt"

python -u vla-scripts/evaluate_calvin.py \
  --pretrained_checkpoint "$CALVIN_CKPT" \
  --calvin_path "$CALVIN_ROOT" \
  --use_checkpoint_modeling True \
  --validate_checkpoint_weights True \
  --use_minivlm True \
  --use_l1_regression True \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --use_pro_version True \
  --center_crop True \
  --num_open_loop_steps 8 \
  --num_sequences 1000 \
  --warmup_queries 10 \
  --save_videos False \
  --eval_log_dir "$PROJECT_ROOT/eval_logs/calvin" \
  --phase Inference \
  2>&1 | tee "$PROJECT_ROOT/eval_logs/CALVIN_ABC_AsymDepth_eval1000.log"
```

CALVIN results, success rates, and latency metrics are written to `eval_logs/calvin/<timestamp_checkpoint-name>/`.

## Reproduction Notes

- The commands above use a single GPU. The audited training path rejects unverified multi-process training.
- `batch_size=4` with `grad_accumulation_steps=4` gives an effective batch size of 16. For lower-memory GPUs, use `1×16` or `2×8`, with a corresponding change in training speed.
- Each teacher checkpoint must contain the complete VLA weights, `action_head`, `proprio_projector`, and `dataset_statistics.json`.
- The training dataset and teacher checkpoint must use identical action and proprioception normalization statistics.
- LIBERO evaluation uses the default initial states. Different GPUs, CUDA versions, and simulator environments may produce small differences in success rate and latency.
- `outputs/`, `eval_logs/`, datasets, and model weights are excluded by `.gitignore` and will not be committed to GitHub.

## Acknowledgements

This project builds on the open-source implementations of VLA-Adapter, OpenVLA, Prismatic VLM, LIBERO, and CALVIN. We thank the authors and contributors of these projects.

## License

This project follows the `LICENSE` included in this repository.
