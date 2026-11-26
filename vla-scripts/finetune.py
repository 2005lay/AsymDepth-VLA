"""
finetune.py

Fine-tunes Qwen2.5-0.5B via LoRA.
"""

import os
import time
import json
import copy
import random
import shutil
import numpy as np
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type
import torch.nn.functional as F
import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    NUM_TOKENS
)
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.models import load, load_vla
from prismatic.models.pruning import (DEFAULT_24_LAYERS, load_checkpoint_state_dict, parse_kept_layers,
                                      slice_state_dict, load_complete_state_dict)
from prismatic.models.bridge_contract import FEATURE_CONTRACT, extract_bridge_states



# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class FinetuneConfig:
    # fmt: off
    config_file_path: str = "openvla/openvla-7b"     # Path to necessary config files of LA-Adapter
    vlm_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)
    use_minivlm: bool = False                        # 
    resum_vla_path: str = "openvla/openvla-7b"       # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_spline_head: bool = False                    # If True, trains B-spline action head
    use_state_adapter: bool = True                   # If True, uses state-aware modulation in spline head
    num_control_points: int = 4                      # Number of B-spline control points
    adaptive_control_points: bool = True             # If True, enables adaptive control-point gating
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for training 
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input
    phase1_path: str = "None"

    # Bridge-preserving pruning and online recovery distillation
    vlm_kept_layers: str = DEFAULT_24_LAYERS
    policy_kept_layers: str = DEFAULT_24_LAYERS
    enable_gate_pruning: bool = False
    lambda_gate: float = 1e-4
    use_distill: bool = False
    inherit_teacher_weights: bool = True
    student_init_mode: str = "auto"  # auto|teacher_sliced|base_vlm_random_policy|random_selected|fully_random
    save_steps: str = ""  # Explicit optimizer steps; includes optional step 0
    teacher_ckpt: str = "outputs/LIBERO-Spatial-Pro"
    lambda_action: float = 1.0
    lambda_rep: float = 0.5
    seed: int = 7
    check_gradients: bool = True
    verify_checkpoint: bool = True

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100000             # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200000                          # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = False                           # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = False         # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # Full Finetune
    use_fz: bool = False                             # If True, uses LoRA fine-tuning

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # revision version
    use_pro_version: bool = True                             # the version number
    phase: str = "Training"
    # fmt: on



def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


STUDENT_INIT_MODES = {"teacher_sliced", "base_vlm_random_policy", "random_selected", "fully_random"}


def is_random_selected_vlm_parameter(name: str) -> bool:
    """Parameters randomized in the no-teacher-selected-weights ablation.

    The compact Qwen decoder blocks are exactly the VLM layers selected by the
    pruning recipe.  Every non-layer tensor remains identical to the inherited
    teacher-initialized baseline, so the experiment changes one factor only.
    """
    return name.startswith("language_model.model.layers.")


def is_trainable_random_selected_vlm_parameter(name: str) -> bool:
    """VLM tensors updated in random-selected mode.

    Selected compact decoder blocks learn from scratch.  Action queries retain
    their teacher initialization but stay trainable, matching the inherited
    baseline's action-query optimization.
    """
    return is_random_selected_vlm_parameter(name) or name == "action_queries.weight"


def expand_tied_parameter_aliases(module: nn.Module, state_dict: dict) -> dict:
    """Populate missing state-dict aliases that share the same Parameter.

    Prismatic/Qwen checkpoints can serialize only one side of tied token
    embedding/LM-head weights.  Expanding proven aliases lets the subsequent
    completeness check distinguish a legitimate tied omission from a genuinely
    missing pretrained tensor.
    """
    expanded = dict(state_dict)
    aliases = {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for names in aliases.values():
        present = [name for name in names if name in expanded]
        if present:
            for name in names:
                expanded.setdefault(name, expanded[present[0]])
    return expanded


def build_random_selected_vlm_state(
    module: nn.Module,
    teacher_source_state: dict,
    kept_layers,
) -> dict:
    """Combine teacher non-layer tensors with random selected compact layers.

    Decoder blocks in the compact model are taken from ``module`` exactly as
    initialized; every other tensor must come from the matching LIBERO teacher.
    """
    target_state = module.state_dict()
    teacher_sliced = slice_state_dict(teacher_source_state, target_state, kept_layers)
    teacher_sliced = expand_tied_parameter_aliases(module, teacher_sliced)
    nonrandom_keys = [key for key in target_state if not is_random_selected_vlm_parameter(key)]
    missing_teacher = [key for key in nonrandom_keys if key not in teacher_sliced]
    if missing_teacher:
        raise RuntimeError(
            "random_selected requires complete teacher non-selected tensors; "
            f"missing {missing_teacher[:20]}"
        )
    return {
        key: (target_state[key] if is_random_selected_vlm_parameter(key) else teacher_sliced[key])
        for key in target_state
    }


def build_base_vlm_random_policy_state(
    module: nn.Module,
    generic_base_source_state: dict,
    kept_layers,
) -> dict:
    """Initialize the compact VLM from the generic base, never the LIBERO teacher.

    Vision encoders, multimodal projector, token embeddings, selected Qwen
    layers, final norm and LM head are sliced from the generic pretrained VLM.
    The action queries are VLA-specific and remain freshly randomized together
    with the separately constructed Policy/Action Head and proprio projector.
    """
    target_state = module.state_dict()
    base_sliced = slice_state_dict(generic_base_source_state, target_state, kept_layers)
    base_sliced = expand_tied_parameter_aliases(module, base_sliced)
    random_keys = {"action_queries.weight"}
    required_base = [key for key in target_state if key not in random_keys]
    missing_base = [key for key in required_base if key not in base_sliced]
    if missing_base:
        raise RuntimeError(
            "base_vlm_random_policy requires complete generic VLM tensors; "
            f"missing {missing_base[:20]}"
        )
    return {
        key: (target_state[key].detach().cpu().clone() if key in random_keys else base_sliced[key])
        for key in target_state
    }


def load_local_prismatic_vlm_state(model_dir: str | Path) -> dict:
    """Read a complete local Prismatic VLM checkpoint without Hub materialization.

    ``prismatic.models.load`` constructs DINO/SigLIP with ``pretrained=True``
    before applying the local aggregate checkpoint, which unnecessarily reaches
    Hugging Face.  For weight slicing we only need the checkpoint tensors, so
    load and flatten its component state dictionaries directly and stay fully
    offline.
    """
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Local generic VLM directory not found: {root}")
    checkpoint_dir = root / "checkpoints"
    preferred = checkpoint_dir / "step-020792-epoch-01-loss=0.5268.pt"
    if preferred.is_file():
        checkpoint = preferred
    else:
        candidates = sorted(checkpoint_dir.glob("*.pt"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected the canonical checkpoint or one .pt under {checkpoint_dir}, got {candidates}"
            )
        checkpoint = candidates[0]
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise TypeError(f"Invalid Prismatic checkpoint payload: {checkpoint}")
    model_state = payload["model"]
    # Prismatic finetune checkpoints always contain projector + LLM.  Some
    # releases intentionally omit the frozen foundation vision encoders and
    # expect them to be materialized from their official DINO/SigLIP weights.
    required = ("projector", "llm_backbone")
    missing_components = [name for name in required if not isinstance(model_state.get(name), dict)]
    if missing_components:
        raise RuntimeError(f"Generic VLM checkpoint missing components {missing_components}: {checkpoint}")
    flattened = {}
    components = list(required)
    if isinstance(model_state.get("vision_backbone"), dict):
        components.insert(0, "vision_backbone")
    else:
        config_path = root / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Generic VLM config missing: {config_path}")
        model_cfg = json.loads(config_path.read_text(encoding="utf-8")).get("model")
        if not isinstance(model_cfg, dict):
            raise RuntimeError(f"Generic VLM config has no model section: {config_path}")
        from prismatic.models.materialize import get_vision_backbone_and_transform
        print(
            "[CHECK] aggregate VLM checkpoint omits frozen vision weights; "
            "loading official DINO/SigLIP foundation weights from HF cache/endpoint; "
            f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '<default>')}, "
            f"HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET', '<unset>')}"
        )
        vision_backbone, _ = get_vision_backbone_and_transform(
            model_cfg["vision_backbone_id"],
            model_cfg["image_resize_strategy"],
            int(model_cfg.get("image_sequence_len", 1)),
        )
        model_state = dict(model_state)
        model_state["vision_backbone"] = vision_backbone.state_dict()
        components.insert(0, "vision_backbone")
        del vision_backbone
    for component in components:
        for key, value in model_state[component].items():
            if torch.is_tensor(value):
                flattened[f"{component}.{str(key).removeprefix('module.')}"] = value
    if not flattened:
        raise RuntimeError(f"Generic VLM checkpoint contains no tensors: {checkpoint}")
    print(f"[CHECK] loaded generic VLM tensors from local checkpoint + foundation vision source: {checkpoint}")
    return flattened



def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.config_file_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.config_file_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_fz:
            run_id += f"+frozen+dropout-{cfg.lora_dropout}"
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id



def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)



def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)



def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    print(f"# trainable params in {name}: {num_params}")



def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.resum_vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)
        print('loaded!!!!!!!!!')

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)


def build_frozen_teacher(cfg: FinetuneConfig, device_id: int):
    """Build a CPU/GPU-frozen teacher for online action distillation.

    The teacher is loaded only when requested.  It is deliberately kept out of
    the optimizer and all calls are wrapped in ``torch.no_grad`` in the forward
    pass, which prevents teacher activations from being retained for backward.
    """
    if not cfg.use_distill:
        return None, None, None
    if not cfg.teacher_ckpt:
        raise ValueError("--teacher_ckpt is required when --use_distill=True")
    teacher_cfg = type("TeacherCfg", (), {})()
    teacher_cfg.pretrained_checkpoint = cfg.teacher_ckpt
    teacher_cfg.load_in_8bit = False
    teacher_cfg.load_in_4bit = False
    teacher_cfg.use_film = cfg.use_film
    teacher_cfg.num_images_in_input = cfg.num_images_in_input
    teacher_cfg.model_family = "openvla"
    teacher_cfg.vlm_kept_layers = DEFAULT_24_LAYERS
    teacher_cfg.policy_kept_layers = DEFAULT_24_LAYERS
    teacher_cfg.enable_gate_pruning = False
    teacher_cfg.lambda_gate = cfg.lambda_gate
    teacher = load_teacher_for_distillation(teacher_cfg)
    teacher.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    teacher = teacher.to(device_id)
    teacher.eval()
    teacher.requires_grad_(False)
    teacher_head = None
    teacher_proprio = None
    if cfg.use_proprio:
        teacher_proprio = ProprioProjector(llm_dim=teacher.llm_dim, proprio_dim=PROPRIO_DIM).to(device_id).to(torch.bfloat16)
        load_component_checked(
            teacher_proprio,
            load_component_from_dir(cfg.teacher_ckpt, "proprio_projector"),
            "teacher proprio_projector",
        )
        teacher_proprio.eval(); teacher_proprio.requires_grad_(False)
    teacher_head = L1RegressionActionHead(
        input_dim=teacher.llm_dim, hidden_dim=teacher.llm_dim, action_dim=ACTION_DIM,
        use_pro_version=cfg.use_pro_version, policy_kept_layers=DEFAULT_24_LAYERS,
        vlm_kept_layers=DEFAULT_24_LAYERS,
        num_task_tokens=teacher.vision_backbone.get_num_patches() * cfg.num_images_in_input,
    ).to(device_id).to(torch.bfloat16)
    load_component_checked(
        teacher_head,
        load_component_from_dir(cfg.teacher_ckpt, "action_head"),
        "teacher action_head",
    )
    teacher_head.eval(); teacher_head.requires_grad_(False)
    return teacher, teacher_head, teacher_proprio


def load_component_from_dir(checkpoint_dir: str, component: str):
    checkpoint_dir = Path(checkpoint_dir)
    matches = sorted(checkpoint_dir.glob(f"{component}*checkpoint.pt"))
    if not matches:
        raise FileNotFoundError(f"No {component} checkpoint found under {checkpoint_dir}")
    if len(matches) != 1:
        raise RuntimeError(f"Ambiguous {component} files: {matches}; select a single-step checkpoint")
    state = torch.load(matches[0], map_location="cpu", weights_only=True)
    return remove_ddp_in_checkpoint(state)


def load_component_checked(module: nn.Module, state_dict: dict, component: str) -> None:
    """Load a component and fail fast instead of silently training with gaps."""
    load_complete_state_dict(module, state_dict, component)


def load_student_proprio_from_teacher(proprio_projector, cfg) -> None:
    """Warm-start the student proprio projector from the full teacher."""
    state = load_component_from_dir(cfg.teacher_ckpt, "proprio_projector")
    target = proprio_projector.module if hasattr(proprio_projector, "module") else proprio_projector
    load_component_checked(target, state, "student proprio_projector")


def load_teacher_for_distillation(cfg):
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    teacher, info = OpenVLAForActionPrediction.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys"):
        raise RuntimeError(f"Teacher VLM is incomplete/incompatible: {info}")
    if len(teacher.language_model.model.layers) != 24:
        raise ValueError("Distillation requires an unpruned 24-layer teacher")
    return teacher


def load_student_action_head_from_teacher(action_head, cfg) -> None:
    """Initialize a compact policy head by slicing the official 24-block head."""
    teacher_dir = Path(cfg.teacher_ckpt)
    if not cfg.teacher_ckpt or not teacher_dir.exists():
        return
    try:
        source = load_component_from_dir(str(teacher_dir), "action_head")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing teacher action_head under {teacher_dir}") from exc
    target_module = action_head.module if hasattr(action_head, "module") else action_head
    kept = parse_kept_layers(cfg.policy_kept_layers)
    sliced = slice_state_dict(source, target_module.state_dict(), kept)
    load_complete_state_dict(target_module, sliced, "sliced student action_head")
    print(f"Loaded sliced teacher policy weights ({len(kept)} blocks); missing=0, unexpected=0")


def load_student_action_head_with_random_selected_blocks(action_head, cfg) -> None:
    """Inherit policy bridges while leaving selected residual blocks random."""
    source = load_component_from_dir(cfg.teacher_ckpt, "action_head")
    target_module = action_head.module if hasattr(action_head, "module") else action_head
    target_state = target_module.state_dict()
    sliced = slice_state_dict(source, target_state, parse_kept_layers(cfg.policy_kept_layers))
    random_prefix = "model.mlp_resnet_blocks."
    nonrandom_keys = [key for key in target_state if not key.startswith(random_prefix)]
    missing_teacher = [key for key in nonrandom_keys if key not in sliced]
    if missing_teacher:
        raise RuntimeError(
            "random_selected policy requires complete teacher bridge tensors; "
            f"missing {missing_teacher[:20]}"
        )
    mixed = {
        key: (target_state[key] if key.startswith(random_prefix) else sliced[key])
        for key in target_state
    }
    load_complete_state_dict(target_module, mixed, "teacher policy bridges plus random selected blocks")
    print(
        f"[CHECK] RANDOM selected policy blocks={len(parse_kept_layers(cfg.policy_kept_layers))}; "
        "teacher fc1/fc2 bridges retained"
    )



def run_forward_pass(
    vla,
    action_head,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_spline_head,
    use_proprio,
    use_film,
    num_patches,
    compute_diffusion_l1=False,
    use_pro_version=True,
    cfg=None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_spline_head (bool): Whether to use spline-based continuous decoding.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps (int): Number of diffusion steps (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"].to(device_id),
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=use_film,
            )

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:,1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression or use_spline_head):
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)

        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=current_action_mask
            )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, 
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=current_action_mask
            )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=next_actions_mask
            )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, 
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=next_actions_mask
            )
        
        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )
        
    # Compute metrics for continuous action representations (L1 regression)
    else:
        multi_layer_hidden_states = extract_bridge_states(
            output.hidden_states, batch["labels"].to(device_id), num_patches, NUM_TOKENS
        )

        # Enter DDP via forward(), including both trainable policy and proprio.
        predicted_actions = action_head(
            multi_layer_hidden_states,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            phase=getattr(cfg, "phase", "Training"),
            )

        loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)

        # Optional online action/representation distillation.  The teacher is
        # attached by ``finetune`` in eval/no-grad mode and never enters the
        # student's autograd graph.
        if cfg is not None and getattr(cfg, "use_distill", False) and getattr(cfg, "teacher_model", None) is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_output = cfg.teacher_model(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"].to(device_id),
                    output_hidden_states=True,
                    proprio=batch["proprio"] if use_proprio else None,
                    proprio_projector=getattr(cfg, "teacher_proprio_projector", None),
                    noisy_actions=None,
                    noisy_action_projector=None,
                    diffusion_timestep_embeddings=None,
                    use_film=use_film,
                )
                teacher_states = extract_bridge_states(
                    teacher_output.hidden_states, batch["labels"].to(device_id), num_patches, NUM_TOKENS
                )
                teacher_actions = cfg.teacher_action_head.predict_action(
                    teacher_states,
                    proprio=batch["proprio"] if use_proprio else None,
                    proprio_projector=getattr(cfg, "teacher_proprio_projector", None),
                    phase="Inference",
                )
            action_distill = F.mse_loss(predicted_actions.float(), teacher_actions.float())
            student_top = multi_layer_hidden_states[:, -1, num_patches:, :]
            teacher_top = teacher_states[:, -1, num_patches:, :]
            rep_distill = F.mse_loss(student_top.float(), teacher_top.float())
            loss = loss + cfg.lambda_action * action_distill + cfg.lambda_rep * rep_distill
            metrics.update({"distill_action_loss": action_distill.item(), "distill_rep_loss": rep_distill.item()})

        unwrapped_head = getattr(action_head, "module", action_head)
        if getattr(cfg, "enable_gate_pruning", False) and hasattr(unwrapped_head, "gate_l1_loss"):
            gate_loss = unwrapped_head.gate_l1_loss().to(loss.dtype)
            loss = loss + cfg.lambda_gate * gate_loss
            metrics["gate_l1_loss"] = gate_loss.item()

        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
            }
        )

        # Get detailed L1 losses for logging
        should_log_l1_loss = use_l1_regression or use_spline_head
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            if compute_diffusion_l1:
                print('curr: ',curr_action_l1_loss.item())
                # print('next: ',next_actions_l1_loss.item())

            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    return loss, metrics



def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics



def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)



def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    distributed_state,
    new_state_dict,
    
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.

    Returns:
        None.
    """
    checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
    if checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to overwrite step checkpoint: {checkpoint_dir}")
    # LoRA runs are reconstructed and merged on CPU without touching the LIVE
    # BF16 model.  Full-random runs have no adapter and save the live full
    # student directly; both paths produce the same directly evaluable HF
    # checkpoint layout.
    cpu_rng = torch.get_rng_state()
    try:
        if distributed_state.is_main_process:
            checkpoint_dir.mkdir(parents=True)
            suffix = f"{log_step}_checkpoint.pt"
            processor.save_pretrained(checkpoint_dir)
            save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
            live = vla.module
            for name, module in (("action_head", action_head), ("proprio_projector", proprio_projector)):
                torch.save(module.module.state_dict(), checkpoint_dir / f"{name}--{suffix}")
            if cfg.use_lora:
                adapter_dir = checkpoint_dir / "lora_adapter"
                live.save_pretrained(adapter_dir)
                base = OpenVLAForActionPrediction(copy.deepcopy(live.config)).to(torch.bfloat16)
                merge_base = dict(new_state_dict)
                merge_base["action_queries.weight"] = live.get_base_model().action_queries.weight.detach().cpu()
                load_complete_state_dict(base, merge_base, "checkpoint base")
                reconstructed = PeftModel.from_pretrained(base, adapter_dir)
                if cfg.verify_checkpoint:
                    live_state = live.state_dict()
                    for key, value in reconstructed.state_dict().items():
                        if key not in live_state or not torch.equal(value.cpu(), live_state[key].detach().cpu()):
                            raise RuntimeError(f"Checkpoint reconstruction differs from LIVE model: {key}")
                    print("[CHECK] reconstructed unmerged weights match LIVE tensors")
                inference_model = reconstructed.merge_and_unload(safe_merge=True)
                # A .bin stores every tied alias; outer HF wrappers do not always
                # advertise nested Qwen tied keys to the safetensors serializer.
                inference_model.save_pretrained(checkpoint_dir, safe_serialization=False)
            else:
                if hasattr(live, "peft_config"):
                    raise RuntimeError("Full-training checkpoint unexpectedly contains a PEFT wrapper")
                inference_model = live
                inference_model.save_pretrained(checkpoint_dir, safe_serialization=False)
            hf_dir = Path(__file__).resolve().parents[1] / "prismatic" / "extern" / "hf"
            for name in ("configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py"):
                shutil.copy2(hf_dir / name, checkpoint_dir / name)
            config_path = checkpoint_dir / "config.json"
            metadata = json.loads(config_path.read_text(encoding="utf-8"))
            init_mode = getattr(cfg, "student_init_mode", "auto")
            if init_mode == "auto":
                init_mode = "teacher_sliced" if getattr(cfg, "inherit_teacher_weights", True) else "fully_random"
            if cfg.use_lora:
                training_mode = "lora"
            elif init_mode == "random_selected":
                training_mode = "selected_layers_full_finetune"
            else:
                training_mode = "full_parameter"
            if init_mode == "teacher_sliced":
                initialization_label = "teacher"
            elif init_mode == "base_vlm_random_policy":
                initialization_label = "generic_base_vlm_random_policy"
            else:
                initialization_label = "random"
            metadata.update({
                "vlm_kept_layers": cfg.vlm_kept_layers, "policy_kept_layers": cfg.policy_kept_layers,
                "student_initialization": initialization_label,
                "student_init_mode": init_mode,
                "student_training_mode": training_mode,
                "student_weight_sources": ({
                    "vlm": "generic_pretrained_base",
                    "action_queries": "random",
                    "policy_action_head": "random",
                    "proprio_projector": "random",
                    "libero_teacher": "distillation_only",
                } if init_mode == "base_vlm_random_policy" else None),
                "feature_contract": FEATURE_CONTRACT, "policy_routing": "floor_reindex",
                "num_images_in_input": cfg.num_images_in_input, "use_pro_version": cfg.use_pro_version,
                "policy_hidden_state_indices": getattr(action_head.module, "policy_hidden_indices", None),
                "enable_gate_pruning": cfg.enable_gate_pruning, "lambda_gate": cfg.lambda_gate,
                "norm_stats": json.loads((checkpoint_dir / "dataset_statistics.json").read_text()),
                "distillation": {"enabled": cfg.use_distill, "teacher_ckpt": cfg.teacher_ckpt,
                                 "lambda_action": cfg.lambda_action, "lambda_rep": cfg.lambda_rep,
                                 "representation": "top64_official_predecessor_states"},
                "auto_map": {"AutoConfig": "configuration_prismatic.OpenVLAConfig",
                             "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction"},
            })
            config_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            processor_config = {"processor_class": "PrismaticProcessor", "auto_map": {
                "AutoProcessor": "processing_prismatic.PrismaticProcessor"}}
            (checkpoint_dir / "processor_config.json").write_text(json.dumps(processor_config, indent=2), encoding="utf-8")
            preprocessor_path = checkpoint_dir / "preprocessor_config.json"
            preprocessor_data = json.loads(preprocessor_path.read_text(encoding="utf-8"))
            preprocessor_data["auto_map"] = {"AutoImageProcessor": "processing_prismatic.PrismaticImageProcessor",
                                             "AutoProcessor": "processing_prismatic.PrismaticProcessor"}
            preprocessor_path.write_text(json.dumps(preprocessor_data, indent=2), encoding="utf-8")
            if cfg.verify_checkpoint:
                disk = load_checkpoint_state_dict(checkpoint_dir)
                expected = inference_model.state_dict()
                for key, value in disk.items():
                    if key not in expected or not torch.equal(value, expected[key].detach().cpu()):
                        raise RuntimeError(f"VLM disk readback differs: {key}")
                    if not torch.isfinite(value).all().item():
                        raise RuntimeError(f"Non-finite saved tensor: {key}")
                if cfg.use_lora:
                    load_complete_state_dict(inference_model, disk, "disk readback completeness")
                else:
                    # Never load back into the live training model.  Verify
                    # completeness against an independent CPU instance.
                    verifier = OpenVLAForActionPrediction(copy.deepcopy(live.config)).to(torch.bfloat16)
                    load_complete_state_dict(verifier, disk, "disk readback completeness")
                    del verifier
                for name, module in (("action_head", action_head), ("proprio_projector", proprio_projector)):
                    state = load_component_from_dir(str(checkpoint_dir), name)
                    for key, value in module.module.state_dict().items():
                        if not torch.equal(state[key], value.detach().cpu()):
                            raise RuntimeError(f"Component disk readback differs: {name}.{key}")
                print("[CHECK] disk VLM and component readback passed")
            (checkpoint_dir / "checkpoint_complete.json").write_text(json.dumps({
                "step": log_step, "full_inference_weights": True,
                "full_trainable_student_weights": True,
                "optimizer_state_saved": False,
                "tensor_readback_checked": cfg.verify_checkpoint, "feature_contract": FEATURE_CONTRACT,
            }, indent=2), encoding="utf-8")
            print(f"[CHECK] FULL checkpoint COMPLETE: {checkpoint_dir}")
            if cfg.use_lora:
                del reconstructed, base, merge_base
            del inference_model
        dist.barrier()
    finally:
        torch.set_rng_state(cpu_rng)



def run_validation(
    vla,
    action_head,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    modules = [module for module in (vla, action_head, proprio_projector) if module is not None]
    modes = [module.training for module in modules]
    validation_cfg = copy.copy(cfg)
    validation_cfg.phase = "Inference"  # No training perturbations in validation.
    all_val_metrics = []
    try:
        for module in modules:
            module.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                _, metrics = run_forward_pass(
                    vla=vla, action_head=action_head, proprio_projector=proprio_projector,
                    batch=batch, action_tokenizer=action_tokenizer, device_id=device_id,
                    use_l1_regression=cfg.use_l1_regression, use_spline_head=cfg.use_spline_head,
                    use_proprio=cfg.use_proprio, use_film=cfg.use_film, num_patches=num_patches,
                    compute_diffusion_l1=False, use_pro_version=cfg.use_pro_version, cfg=validation_cfg,
                )
                metrics["loss"] = metrics["loss_value"]
                all_val_metrics.append(metrics)
                if time.time() - val_start_time > val_time_limit:
                    break
    finally:
        for module, was_training in zip(modules, modes):
            module.train(was_training)
    if not all_val_metrics:
        raise RuntimeError("Validation dataloader yielded no batches")
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0]:
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        avg_val_metrics[metric_name] = sum(values) / len(values)
    avg_val_metrics["val_batches_count"] = len(all_val_metrics)
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """ 

    global RAW_STATE_DICT
    # This audited path is intentionally specific; do not silently route an
    # unsupported spline/diffusion/resume experiment through the L1 trainer.
    if not (cfg.use_minivlm and cfg.use_l1_regression and cfg.use_proprio and cfg.use_pro_version):
        raise ValueError("This trainer requires minivlm + Pro + L1 regression + proprio")
    if cfg.use_spline_head or cfg.use_diffusion or cfg.use_film or cfg.use_fz or cfg.resume:
        raise ValueError("Spline/diffusion/FiLM/full-finetune/resume are not supported in this audited run")
    if cfg.save_latest_checkpoint_only:
        raise ValueError("This audited run requires save_latest_checkpoint_only=False")
    init_mode = str(cfg.student_init_mode).strip().lower()
    if init_mode == "auto":
        # Backward compatibility for previously delivered commands.  Audited
        # ablations must request their explicit mode instead of relying on auto.
        init_mode = "teacher_sliced" if cfg.inherit_teacher_weights else "fully_random"
    if init_mode not in STUDENT_INIT_MODES:
        raise ValueError(f"student_init_mode must be one of {sorted(STUDENT_INIT_MODES)}, got {init_mode!r}")
    cfg.student_init_mode = init_mode
    if cfg.student_init_mode == "teacher_sliced" and not cfg.inherit_teacher_weights:
        raise ValueError("teacher_sliced requires inherit_teacher_weights=True")
    if cfg.student_init_mode != "teacher_sliced" and cfg.inherit_teacher_weights:
        raise ValueError(f"{cfg.student_init_mode} requires inherit_teacher_weights=False")

    if cfg.student_init_mode in {"teacher_sliced", "base_vlm_random_policy"}:
        if not cfg.use_lora or not cfg.merge_lora_during_training:
            mode_description = (
                "Teacher-initialized recovery" if cfg.student_init_mode == "teacher_sliced"
                else "Generic-base-VLM/random-Policy recovery"
            )
            raise ValueError(f"{mode_description} requires use_lora=True and merge_lora_during_training=True")
    else:
        # A randomly initialized backbone must itself be optimized.  Applying
        # PEFT here would freeze the random base and train only LoRA adapters,
        # which is not the requested random-initialization training control.
        if cfg.use_lora or cfg.merge_lora_during_training:
            raise ValueError(
                "Random student training requires use_lora=False and "
                "merge_lora_during_training=False"
            )
    if min(cfg.max_steps, cfg.save_freq, cfg.batch_size, cfg.grad_accumulation_steps, cfg.wandb_log_freq) < 1:
        raise ValueError("Steps, batch size, accumulation and save frequency must be positive")
    if cfg.phase != "Training":
        raise ValueError("Fine-tuning must use phase=Training")
    if min(cfg.lambda_action, cfg.lambda_rep, cfg.lambda_gate, cfg.lr_warmup_steps) < 0:
        raise ValueError("Loss coefficients and warmup steps cannot be negative")
    cfg.vlm_kept_layers = ",".join(map(str, parse_kept_layers(cfg.vlm_kept_layers)))
    requested_save_steps = {int(s.strip()) for s in cfg.save_steps.split(",") if s.strip()}
    if any(s < 0 or s > cfg.max_steps for s in requested_save_steps):
        raise ValueError("save_steps must be between 0 and max_steps")
    cfg.policy_kept_layers = ",".join(map(str, parse_kept_layers(cfg.policy_kept_layers)))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    import tensorflow as tf
    tf.random.set_seed(cfg.seed)
    tf.config.set_visible_devices([], "GPU")  # The RLDS loader must not reserve PyTorch GPU memory.

    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.config_file_path = cfg.config_file_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.config_file_path}` on `{cfg.dataset_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # This delivered workflow is single-GPU. Reject an untested distributed
    # launch before directory creation instead of racing on mkdir/checkpoints.
    distributed_state = PartialState()
    if distributed_state.num_processes != 1:
        raise ValueError("This audited workflow requires --nproc_per_node=1")

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Refusing to mix an old experiment with a new run: {run_dir}")
    os.makedirs(run_dir)
    run_configuration = {name: getattr(cfg, name) for name in cfg.__dataclass_fields__}
    (run_dir / "training_config.json").write_text(
        json.dumps(run_configuration, indent=2, default=str), encoding="utf-8"
    )

    # GPU setup
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Initialize wandb logging
    if distributed_state.is_main_process:
        wandb.init(project=cfg.wandb_project, name=f"ft+{run_id}", mode="offline")

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Optional full-depth Teacher source.  When present, its merged HF weights
    # are the authoritative source for compact-subset inheritance (including
    # the fine-tuned VLM), while the local Prismatic VLM remains a fallback.
    teacher_source_state = None
    vlm = None
    if cfg.student_init_mode in {"teacher_sliced", "random_selected"} and cfg.teacher_ckpt and Path(cfg.teacher_ckpt).is_dir():
        try:
            teacher_source_state = load_checkpoint_state_dict(cfg.teacher_ckpt)
            print(f"Loaded Teacher source weights from {cfg.teacher_ckpt} for subset inheritance")
        except (FileNotFoundError, TypeError, RuntimeError) as exc:
            raise RuntimeError("Cannot load the specified teacher; refusing silent base-VLM fallback") from exc
    elif cfg.use_distill and not Path(cfg.teacher_ckpt).is_dir():
        raise FileNotFoundError(f"Full local teacher checkpoint not found: {cfg.teacher_ckpt}")

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect

    if model_is_on_hf_hub(cfg.config_file_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.config_file_path)
        # Overwrite VLA path
        cfg.config_file_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


    # Input checkpoints are read-only. Use registered local classes explicitly
    # during training; package the corresponding code in NEW output checkpoints.
    dist.barrier()

    # Load processor and VLA
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = PrismaticProcessor(
        image_processor=PrismaticImageProcessor.from_pretrained(cfg.config_file_path),
        tokenizer=AutoTokenizer.from_pretrained(cfg.config_file_path, trust_remote_code=False),
    )

    if cfg.use_minivlm:
        hf_token = ''
        needs_base_vlm = cfg.student_init_mode == "teacher_sliced" and teacher_source_state is None
        if needs_base_vlm and 'prism-qwen25-extra-dinosiglip-224px-0_5b' in cfg.vlm_path:
            
            vlm = load(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
        elif needs_base_vlm:
            vlm = load_vla(
                cfg.vlm_path,
                hf_token=hf_token,
                load_for_training=True,
                )
        config = OpenVLAConfig.from_pretrained(cfg.teacher_ckpt if Path(cfg.teacher_ckpt).is_dir()
                                              else cfg.config_file_path)
        if config.text_config.num_hidden_layers != 24:
            raise ValueError("Weight slicing must start from a full 24-layer teacher")
        config.vlm_kept_layers = str(cfg.vlm_kept_layers)
        config.policy_kept_layers = str(cfg.policy_kept_layers)
        config.enable_gate_pruning = bool(cfg.enable_gate_pruning)
        config.lambda_gate = float(cfg.lambda_gate)
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(device_id)  # Create a new model with configuration, the parameters are randomly initialized
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.shape}")
        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
            ]

        def rename_state_dict_keys(state_dict, replace_map):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for old, new in replace_map:
                    if old in new_k:
                        new_k = new_k.replace(old, new)
                new_state_dict[new_k] = v
            return new_state_dict
        
        if cfg.student_init_mode == "base_vlm_random_policy":
            old_state_dict = load_local_prismatic_vlm_state(cfg.vlm_path)
        else:
            old_state_dict = vlm.state_dict() if vlm is not None else {}
        base_source_state = rename_state_dict_keys(old_state_dict, replace_map)
        if cfg.student_init_mode == "teacher_sliced":
            inheritance_source = dict(base_source_state)
            if teacher_source_state is not None:
                inheritance_source.update(teacher_source_state)
            RAW_STATE_DICT = slice_state_dict(
                inheritance_source,
                vla.state_dict(),
                parse_kept_layers(cfg.vlm_kept_layers),
            )
            load_complete_state_dict(vla, RAW_STATE_DICT, "sliced student VLM")
            print(f"[CHECK] inherited all tensors for {len(vla.vlm_kept_layers)} physical VLM layers")
            del inheritance_source
        elif cfg.student_init_mode == "base_vlm_random_policy":
            # The student receives no LIBERO teacher weights.  Its 16-layer VLM
            # comes only from the generic multimodal base; action queries and
            # the separately built Policy/proprio modules start randomly.
            RAW_STATE_DICT = build_base_vlm_random_policy_state(
                vla,
                base_source_state,
                parse_kept_layers(cfg.vlm_kept_layers),
            )
            load_complete_state_dict(vla, RAW_STATE_DICT, "generic base VLM plus random action queries")
            print(
                f"[CHECK] GENERIC BASE VLM sliced to {len(vla.vlm_kept_layers)} physical layers; "
                "random action queries; zero LIBERO teacher tensors copied into student"
            )
        elif cfg.student_init_mode == "random_selected":
            # Preserve every baseline tensor except the compact decoder blocks.
            # This isolates selected-layer teacher initialization as the only
            # changed factor in the ablation.
            mixed_state = build_random_selected_vlm_state(
                vla,
                teacher_source_state,
                parse_kept_layers(cfg.vlm_kept_layers),
            )
            target_state = vla.state_dict()
            load_complete_state_dict(vla, mixed_state, "teacher bridges plus random selected VLM layers")
            random_count = sum(is_random_selected_vlm_parameter(key) for key in target_state)
            RAW_STATE_DICT = {}
            print(
                f"[CHECK] RANDOM selected VLM tensors={random_count}; "
                f"teacher non-layer tensors retained={len(target_state) - random_count}; "
                "no selected VLM layer weights copied"
            )
            del mixed_state, target_state
        else:
            # Full-parameter checkpoints are saved directly from the live
            # student, so retaining a second CPU copy of the random base would
            # only waste host memory.
            RAW_STATE_DICT = {}
            print("[CHECK] FULLY RANDOM student VLM; no pretrained tensors loaded")
        del old_state_dict, base_source_state
        del teacher_source_state, vlm

    else:
        RAW_STATE_DICT ={}
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=False,
            ).to(device_id)

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.config.feature_contract = FEATURE_CONTRACT
    vla.config.policy_routing = "floor_reindex"
    vla.config.num_images_in_input = cfg.num_images_in_input

    # vla.set_version(cfg.version)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha= 2 * cfg.lora_rank,
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True
        vla.print_trainable_parameters()

    else:
        if cfg.student_init_mode == "random_selected":
            # Optimize every randomized decoder block.  Inherited action
            # queries remain trainable just as in the teacher-sliced baseline.
            vla.requires_grad_(False)
            vla.language_model.model.layers.requires_grad_(True)
            vla.action_queries.requires_grad_(True)
            unexpected_trainable = [
                name for name, parameter in vla.named_parameters()
                if parameter.requires_grad and not is_trainable_random_selected_vlm_parameter(name)
            ]
            missing_trainable = [
                name for name, parameter in vla.named_parameters()
                if is_trainable_random_selected_vlm_parameter(name) and not parameter.requires_grad
            ]
            if unexpected_trainable or missing_trainable:
                raise RuntimeError(
                    "random_selected trainability mismatch: "
                    f"unexpected={unexpected_trainable[:10]}, missing={missing_trainable[:10]}"
                )
            print(
                "[CHECK] selected random VLM blocks train from scratch; inherited action queries remain trainable; "
                "inherited vision/projector/embedding/norm frozen"
            )
        else:
            # Strict full-scratch control: optimize every VLA parameter.
            vla.requires_grad_(True)
            frozen = [name for name, param in vla.named_parameters() if not param.requires_grad]
            if frozen:
                raise RuntimeError(f"Random student unexpectedly contains frozen parameters: {frozen[:10]}")
            print(f"[CHECK] FULL random VLM training; trainable tensors={len(list(vla.parameters()))}")
        vla.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        vla.config.use_cache = False
        vla.config.text_config.use_cache = False
        vla.language_model.config.use_cache = False

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For this, must specify `vla.model.vision_backbone` instead of just `vla.vision_backbone`, since the
        # latter would cause the new wrapped backbone to be saved as a new attribute of `vla` instead of overwriting the
        # original one (due to the LoRA wrapper)
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.config_file_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # If applicable, instantiate proprio projector
    action_head, proprio_projector = None, None
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
            to_bf16=True,
        )
        if cfg.student_init_mode in {"teacher_sliced", "random_selected"} and cfg.teacher_ckpt and Path(cfg.teacher_ckpt).is_dir():
            load_student_proprio_from_teacher(proprio_projector, cfg)

    # If applicable, instantiate continuous action head for L1 regression or spline head
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "use_pro_version": cfg.use_pro_version,
                "num_task_tokens": vla.module.vision_backbone.get_num_patches() * cfg.num_images_in_input,
                "policy_kept_layers": cfg.policy_kept_layers,
                "vlm_kept_layers": cfg.vlm_kept_layers,
                "enable_gate_pruning": cfg.enable_gate_pruning,
                "lambda_gate": cfg.lambda_gate,
            },
            to_bf16=True,
            find_unused_params=True,
        )

    # Build the frozen full-depth teacher only after the student components are
    # initialized, so it is never included in the optimizer parameter list.
    if action_head is not None:
        if cfg.student_init_mode == "teacher_sliced":
            load_student_action_head_from_teacher(action_head, cfg)
        elif cfg.student_init_mode == "random_selected":
            load_student_action_head_with_random_selected_blocks(action_head, cfg)
            print("[CHECK] teacher proprio/action-query/Policy bridges retained; selected Policy blocks random")
        else:
            print("[CHECK] RANDOM policy/proprio; teacher retained for distillation")
        teacher_model, teacher_action_head, teacher_proprio_projector = build_frozen_teacher(cfg, device_id)
        cfg.teacher_model = teacher_model
        cfg.teacher_action_head = teacher_action_head
        cfg.teacher_proprio_projector = teacher_proprio_projector

    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    print(f"[CHECK] physical policy blocks={len(action_head.module.model.mlp_resnet_blocks)}; "
          f"conditioning hidden indices={action_head.module.policy_hidden_indices}; feature_contract={FEATURE_CONTRACT}")
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings

    # Instantiate optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_spline_head:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]

    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    # 1. MultiStepLR
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
        gamma=0.1,  # Multiplicative factor of learning rate decay
    )
    # 2. CosineAnnealingLR
    # scheduler = CosineAnnealingLR(
    #         optimizer,
    #         T_max=cfg.num_steps_before_decay, 
    #         eta_min=0.0001,          
    #         )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # train_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder,
    # )
    # ---

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1

    # Create training and optional validation datasets
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        use_minivlm=cfg.use_minivlm
        )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_distill:
        stats_path = Path(cfg.teacher_ckpt) / "dataset_statistics.json"
        teacher_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        student_stats = train_dataset.dataset_statistics
        if cfg.dataset_name not in teacher_stats:
            raise ValueError(f"Teacher statistics do not contain {cfg.dataset_name}")
        for group in ("action", "proprio"):
            for key in ("q01", "q99", "mask"):
                left = teacher_stats[cfg.dataset_name][group].get(key)
                right = student_stats[cfg.dataset_name][group].get(key)
                if key == "mask":
                    left = np.ones(len(teacher_stats[cfg.dataset_name][group]["q01"]), dtype=bool) if left is None else left
                    right = np.ones(len(student_stats[cfg.dataset_name][group]["q01"]), dtype=bool) if right is None else right
                if left is None and right is None:
                    continue
                if left is None or right is None or not np.allclose(left, right, atol=1e-5, rtol=1e-5):
                    raise ValueError(f"Teacher/student normalization mismatch: {group}.{key}. "
                                     "Distillation must use the same action/proprio units")
        print("[CHECK] Teacher/student action and proprio normalization agree")
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator and dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    print('Len of dataloader: ', len(dataloader))
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "distill_action_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "distill_rep_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "gate_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    if 0 in requested_save_steps:
        save_training_checkpoint(cfg, run_dir, 0, vla, processor, proprio_projector,
                                 None, action_head, train_dataset, distributed_state, RAW_STATE_DICT)

    # Start training
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            # Compute training metrics and loss
            compute_diffusion_l1 = (cfg.use_l1_regression and batch_idx % cfg.diffusion_sample_freq == 0) or (cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0)
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_spline_head=cfg.use_spline_head,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                compute_diffusion_l1=compute_diffusion_l1,
                use_pro_version=cfg.use_pro_version,
                cfg=cfg,
            )

            # Normalize loss to account for gradient accumulation
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite loss at microbatch {batch_idx}: {metrics}")
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Store recent train metrics
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            if (batch_idx + 1) % cfg.grad_accumulation_steps:
                continue
            # Count COMPLETED optimizer updates, never microbatch groups.
            gradient_step_idx = (batch_idx + 1) // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)
                print(f"[TRAIN] step={log_step}/{cfg.max_steps} {smoothened_metrics}", flush=True)
                with (run_dir / "train_metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": log_step, **smoothened_metrics}) + "\n")

            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            if cfg.lr_warmup_steps > 0 and gradient_step_idx <= cfg.lr_warmup_steps:
                lr_progress = min(gradient_step_idx / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # Log the learning rate
                # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                wandb.log(
                    {
                        "VLA Train/Learning Rate": scheduler.get_last_lr()[0],
                    },
                    step=log_step,
                )

            # Optimizer and LR scheduler step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                if cfg.check_gradients and (gradient_step_idx == 1 or gradient_step_idx % 1000 == 0):
                    for name, module in (("VLM", vla), ("policy", action_head), ("proprio", proprio_projector)):
                        grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
                        if not grads or not any(g.count_nonzero().item() for g in grads):
                            raise RuntimeError(f"No nonzero gradients in {name}")
                        if not all(torch.isfinite(g).all().item() for g in grads):
                            raise FloatingPointError(f"Non-finite gradients in {name}")
                    base = vla.module.get_base_model() if cfg.use_lora else vla.module
                    if base.action_queries.weight.grad is None:
                        raise RuntimeError("Action queries are disconnected from the loss")
                    if cfg.student_init_mode == "fully_random":
                        for component_name in ("vision_backbone", "projector", "language_model"):
                            component = getattr(base, component_name)
                            component_grads = [
                                p.grad for p in component.parameters()
                                if p.requires_grad and p.grad is not None
                            ]
                            if not component_grads or not any(g.count_nonzero().item() for g in component_grads):
                                raise RuntimeError(
                                    f"Random student component has no nonzero gradients: {component_name}"
                                )
                    elif cfg.student_init_mode == "random_selected":
                        layer_grads = [
                            p.grad for p in base.language_model.model.layers.parameters()
                            if p.requires_grad and p.grad is not None
                        ]
                        if not layer_grads or not any(g.count_nonzero().item() for g in layer_grads):
                            raise RuntimeError("Random selected VLM layers have no nonzero gradients")
                        leaked = [
                            name for name, parameter in base.named_parameters()
                            if not is_trainable_random_selected_vlm_parameter(name) and parameter.requires_grad
                        ]
                        if leaked:
                            raise RuntimeError(f"Frozen generic VLM tensors became trainable: {leaked[:10]}")
                    if cfg.use_distill and any(p.grad is not None or p.requires_grad for p in cfg.teacher_model.parameters()):
                        raise RuntimeError("Teacher must be frozen")
                    print(f"[CHECK] gradients VLM/policy/proprio/queries; teacher frozen (step {log_step})")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if (log_step in requested_save_steps if requested_save_steps else log_step % cfg.save_freq == 0) or log_step == cfg.max_steps:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=None,
                    action_head=action_head,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                    new_state_dict=RAW_STATE_DICT,
                )

            # Test model on validation set
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                )
                # Set model back to training mode after validation
                vla.train()

            # Stop training when max_steps is reached
            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
