"""Code to evaluate Calvin."""
import argparse
import hashlib
import json
import logging
import os
# os.environ['PYTHONPATH'] = '/root/RoboDual:' + os.environ.get('PYTHONPATH', '')
from collections import deque
# from peft import PeftModel
from pathlib import Path
# import sys
# sys.path.insert(0, '/root/RoboDual')
import time
import copy
from moviepy.editor import ImageSequenceClip
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
# from openvla.prismatic.vla.action_tokenizer import ActionTokenizer
# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
# sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from vla_evaluation import DualSystemCalvinEvaluation
from calvin_metrics import (
    CalvinRuntimeMetrics,
    build_metrics_payload,
    get_cuda_memory_stats,
    parameter_summary,
    print_markdown_summary,
    reset_cuda_peak_memory_stats,
    write_metrics_json,
)

from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
# from ema_pytorch import EMA
from transformers.modeling_outputs import CausalLMOutputWithPast
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
# logger = logging.getLogger(__name__)

os.environ["FFMPEG_BINARY"] = "auto-detect"
# ``CALVIN_ROOT`` is supplied by the launch command and points at the CALVIN
# source checkout (the directory that contains ``calvin_models/conf``).  Do
# not overwrite it with the relative path ``calvin``: that made the evaluator
# ignore the user's environment and fail as soon as it was launched elsewhere.
CALVIN_ROOT = os.environ.get("CALVIN_ROOT")

from collections import Counter
import json
import numpy as np
from typing import Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import draccus

import os
import torch



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _print_runtime_sources() -> None:
    """Print the exact Python files imported by this process."""
    import calvin_env_wrapper as env_wrapper_module
    import experiments.robot.openvla_utils as openvla_utils_module
    import prismatic.models.action_heads as action_heads_module
    import vla_evaluation as evaluation_module

    print("[CALVIN] runtime source files:")
    current_file = Path(__file__).resolve()
    for label, module in (
        ("evaluate_calvin", None),
        ("calvin_env_wrapper", env_wrapper_module),
        ("vla_evaluation", evaluation_module),
        ("openvla_utils", openvla_utils_module),
        ("action_heads", action_heads_module),
    ):
        path = current_file if module is None else Path(module.__file__).resolve()
        print(f"  {label}: {path}")

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "../outputs/calvin-abc"     # Pretrained checkpoint path

    # Keep the spelling used by the training script.  ``use_minivla`` is a
    # legacy typo and is intentionally not used for CALVIN evaluation.
    use_minivlm: bool = True                    # Official Pro checkpoints use MiniVLM/Qwen prompts
    use_checkpoint_modeling: bool = True        # Official CALVIN checkpoint ships the exact model implementation
    validate_checkpoint_weights: bool = True    # Reject incomplete metadata-only checkpoint directories

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_spline_head: bool = False                    # If True, uses spline-based continuous action head
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    use_x0_prediction: bool = False
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                          # Deterministic 0.9 crop matching training image_aug
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    # task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    # env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)
#################################################################################################################
    # CALVIN
    #################################################################################################################
    calvin_path: str = "calvin"
    num_sequences: int = 1000
    metrics_output: Optional[str] = None
    eval_log_dir: str = "eval_logs/calvin"
    save_videos: bool = False
    warmup_queries: int = 10
    log_dir: str = "log"
    with_depth: bool = True
    with_gripper: bool = True
    with_cfg: bool = True
    enrich_lang: bool = False
    use_pro_version: bool = True
    
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                 # Random Seed (for reproducibility)

    # fmt: on
    save_version: str = "Pro"                        # version of exps

def print_and_save(results, sequences, eval_result_path, task_name=None, epoch=None):
    current_data = {}
    print(f"Results for Epoch {epoch}:")
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

    current_data[epoch] = data

    # Keep the split file next to the result JSON.  The evaluator passes
    # task_name=None, and using ``./None`` here made an unrelated directory in
    # the repository root on every run.
    result_dir = Path(eval_result_path).parent
    result_dir.mkdir(parents=True, exist_ok=True)
    device_index = torch.cuda.current_device() if torch.cuda.is_available() else -1
    with open(result_dir / f"split_{device_index}.json", "w") as file:
        json.dump(chain_sr, file)

    print()
    previous_data = {}
    json_data = {**previous_data, **current_data}
    with open(eval_result_path, "w") as file:
        json.dump(json_data, file)
    print(
        f"Best model: epoch {max(json_data, key=lambda x: json_data[x]['avg_seq_len'])} "
        f"with average sequences length of {max(map(lambda x: x['avg_seq_len'], json_data.values()))}"
    )
    return task_info


def make_env(dataset_path, observation_space, device):
    val_folder = Path(dataset_path) / "validation"
    if not val_folder.is_dir():
        raise FileNotFoundError(
            f"CALVIN validation directory does not exist: {val_folder}. "
            "--calvin_path must contain dataset/task_ABC_D/validation."
        )
    merged_config = val_folder / ".hydra" / "merged_config.yaml"
    if not merged_config.is_file():
        raise FileNotFoundError(
            f"Missing CALVIN merged config: {merged_config}. "
            "Regenerate it from the matching calvin_env config_data_collection.yaml."
        )
    config_text = merged_config.read_text(encoding="utf-8")
    # A file containing only ``env.use_egl: false`` is syntactically valid but
    # unusable: it omits the robot constructor arguments (filename, base pose,
    # joint IDs, etc.) and later fails with a confusing TypeError.  Check for
    # the fields emitted by CALVIN's full Hydra composition up front.
    if "filename:" not in config_text:
        raise ValueError(
            f"CALVIN merged config is incomplete: {merged_config}. "
            "It must contain the expanded robot_cfg (including filename). "
            "Run scripts/export_calvin_merged_config.py against the matching "
            "calvin_env checkout; do not create this file manually."
        )
    robot_fields = (
        "base_pose:",
        "initial_joint_positions:",
        "arm_joint_ids:",
        "gripper_joint_ids:",
        "end_effector_id:",
    )
    if not any(field in config_text for field in robot_fields):
        raise ValueError(
            f"CALVIN merged config is incomplete: {merged_config}. "
            "Robot joint/pose fields are missing; regenerate the full Hydra config."
        )
    from calvin_env_wrapper import CalvinEnvWrapperRaw
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env


def _find_calvin_conf_dir(calvin_root: Optional[str], dataset_root: Optional[str]) -> Path:
    """Find CALVIN's task/annotation configs without conflating code/data roots."""
    candidates = []
    for root in (calvin_root, dataset_root):
        if root:
            root_path = Path(root).expanduser()
            candidates.extend((root_path / "calvin_models" / "conf", root_path / "conf"))
    for candidate in candidates:
        if (candidate / "callbacks/rollout/tasks/new_playtable_tasks.yaml").is_file() and (
            candidate / "annotations/new_playtable_validation.yaml"
        ).is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates) or "<no paths>"
    raise FileNotFoundError(
        "Could not find CALVIN task/annotation configs. Searched: " + searched +
        ". Set CALVIN_ROOT to the CALVIN checkout or pass a dataset root containing calvin_models/conf."
    )


def evaluate_policy(
    model,
    env,
    eval_sr_path,
    eval_result_path,
    num_procs,
    procs_id,
    eval_dir,
    ep_len,
    num_sequences,
    task_name='test',
    enrich_lang=False,
    debug=False,
    calvin_root=None,
    dataset_root=None,
    runtime_metrics=None,
    save_videos=False,
):
    conf_dir = _find_calvin_conf_dir(calvin_root or CALVIN_ROOT, dataset_root)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)


    if enrich_lang:
        with open('/root/RoboDual/vla-scripts/enrich_lang_annotations.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    # val_annotations = {key: val for key, val in val_annotations.items() if 'push' in key and 'right' in key}  # 只保留了push something right的任务！！！！！！！！
    # ``eval_dir`` is already a run-specific directory created by ``main``.
    # CALVIN's helper can append another timestamped directory, which makes
    # metrics difficult to locate and used to place output under the legacy
    # ``evaluation_results`` tree.  Keep all artifacts in the requested run
    # directory instead.
    eval_dir = str(Path(eval_dir).expanduser().resolve())
    Path(eval_dir).mkdir(parents=True, exist_ok=True)
    # Keep a plain list for the final per-task aggregation.  A tqdm iterator
    # is convenient for progress display but should not be reused as the
    # source of task_info after it has been consumed.
    eval_sequences = list(get_sequences(num_sequences))

    num_seq_per_procs = num_sequences // num_procs
    eval_sequences = eval_sequences[num_seq_per_procs * procs_id:num_seq_per_procs * (procs_id + 1)]
    eval_sequences_for_summary = eval_sequences

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env,
            model,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            debug,
            eval_dir,
            sequence_i,
            ep_len,
            runtime_metrics=runtime_metrics,
            save_videos=save_videos,
        )
        results.append(result)
        if not debug:
            success_list = count_success(results)
            with open(eval_sr_path, 'a') as f:
                line = f"{sequence_i}/{num_sequences}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                sequence_i += 1
                line += "\n"
                f.write(line)
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|",
                refresh=False,
            )
        else:
            sequence_i += 1
    task_info = print_and_save(results, eval_sequences_for_summary, eval_result_path, task_name, None)
    return results, task_info


def evaluate_sequence(
    env,
    model,
    task_checker,
    initial_state,
    eval_sequence,
    val_annotations,
    debug,
    eval_dir,
    sequence_i,
    ep_len,
    runtime_metrics=None,
    save_videos=False,
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        if runtime_metrics is not None:
            model.chain_position = subtask_i + 1
        # The released CALVIN evaluator uses three overlapping 8-action
        # chunks and averages actions that target the same environment step.
        # This closed-loop temporal ensemble is important for precise grasping
        # and is the path used for the paper's reported CALVIN score.
        success = rollout_hi3(
            env,
            model,
            task_checker,
            subtask,
            val_annotations,
            debug,
            eval_dir,
            subtask_i,
            sequence_i,
            ep_len,
            save_videos=save_videos,
        )
        if success:  # return 5!!!!!!!!!!!!!!!!!!!
            # print('success: ', subtask_i)
            success_counter += 1
        else:
            return success_counter
    return success_counter


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        # normalized_action[..., -1] = np.sign(normalized_action[..., -1])
        sign = np.sign(normalized_action[..., -1])
        sign = np.array(sign)  # Ensure it is an array and not a scalar
        sign[sign == 0.0] = 1  # Change 0 to 1
        sign[sign == -0.0] = -1  # Change -0 to -1
        normalized_action[..., -1] = sign

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def process_calvin_action(action: np.ndarray) -> np.ndarray:
    """Convert the policy gripper output to CALVIN's binary convention.

    The CALVIN RLDS policy output uses ``0 = close`` and ``1 = open`` for the
    final action dimension.  CALVIN's robot controller uses ``-1 = close``
    and ``+1 = open``.  Keep this conversion in one place so it is not
    accidentally inverted once in the policy wrapper and again here.
    """
    action = np.asarray(action, dtype=np.float32).copy()
    if action.shape[-1] != 7:
        raise ValueError(f"Expected CALVIN action shape (..., 7), got {action.shape}")

    gripper = np.clip(action[..., -1], 0.0, 1.0)
    action[..., -1] = np.where(gripper >= 0.5, 1.0, -1.0)
    return action


def rollout(
    env,
    model,
    task_oracle,
    subtask,
    val_annotations,
    debug,
    eval_dir,
    subtask_i,
    sequence_i,
    ep_len,
    save_videos=False,
):
    # Keep the legacy rollout path subject to the same explicit switch as the
    # CALVIN-Hi3 path.  The paper runs never write videos unless requested.
    _save_rollout_videos.enabled = bool(save_videos)
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    img_dict = {
        'static': [],
        'gripper': [],
    }
    action_queue = deque(maxlen=8)

    for step in range(ep_len):
        if len(action_queue) == 0:
            actions = model.step(obs, lang_annotation, step)
            action_queue.extend(actions)

        action = action_queue.popleft()  # {ndarray: (7,)}
        action = process_calvin_action(action)
        obs, reward, done, current_info = env.step(action.tolist())


        img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
        img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            print(colored("success", "green"), end=" ")
            _save_rollout_videos(img_dict, eval_dir, sequence_i, subtask_i, subtask, "succ")
            return True

    print(colored("fail", "red"), end=" ")
    _save_rollout_videos(img_dict, eval_dir, sequence_i, subtask_i, subtask, "fail")
    return False

import os
import time
import copy
import numpy as np
from moviepy.editor import ImageSequenceClip
from termcolor import colored


def _save_rollout_videos(img_dict, eval_dir, sequence_i, subtask_i, subtask, suffix):
    """Write rollout videos only when explicitly requested."""
    if not getattr(_save_rollout_videos, "enabled", False):
        return
    for key in img_dict.keys():
        clip = ImageSequenceClip(img_dict[key], fps=50)
        clip.write_videofile(
            os.path.join(eval_dir, f"{sequence_i}-{subtask_i}-{subtask}-{key}-{suffix}.mp4"),
            fps=50,
            codec="libx264",
            bitrate="5000k",
            logger=None,
        )


def rollout_hi3(
    env,
    model,
    task_oracle,
    subtask,
    val_annotations,
    debug,
    eval_dir,
    subtask_i,
    sequence_i,
    ep_len,
    save_videos=False,
):
    _save_rollout_videos.enabled = bool(save_videos)
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)

    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    img_dict = {
        'static': [],
        'gripper': [],
    }

    def execute(action):
        nonlocal obs
        action = process_calvin_action(action)
        obs, _, _, current_info = env.step(action.tolist())
        img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
        img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))
        solved = len(task_oracle.get_task_info_for_set(start_info, current_info, {subtask})) > 0
        return solved

    # Match the released evaluator exactly.  Its Hi3 path runs 80 ensemble
    # cycles and intentionally ignores the legacy ep_len argument.
    for _ in range(80):
        action_buffers = [None, None, None]

        action_buffers[0] = model.step(obs, lang_annotation, 0)
        if execute(action_buffers[0][0]):
            print(colored("success", "green"), end=" ")
            _save_rollout_videos(img_dict, eval_dir, sequence_i, subtask_i, subtask, "succ")
            return True

        action_buffers[1] = model.step(obs, lang_annotation, 1)
        if execute((action_buffers[0][1] + action_buffers[1][0]) / 2):
            print(colored("success", "green"), end=" ")
            _save_rollout_videos(img_dict, eval_dir, sequence_i, subtask_i, subtask, "succ")
            return True

        action_buffers[2] = model.step(obs, lang_annotation, 2)
        if execute((action_buffers[0][2] + action_buffers[1][1] + action_buffers[2][0]) / 3):
            print(colored("success", "green"), end=" ")
            _save_rollout_videos(img_dict, eval_dir, sequence_i, subtask_i, subtask, "succ")
            return True

        for t in range(2, 7):
            if execute((action_buffers[0][t] + action_buffers[1][t - 1] + action_buffers[2][t - 2]) / 3):
                print(colored("success", "green"), end=" ")
                _save_rollout_videos(img_dict, eval_dir, sequence_i, subtask_i, subtask, "succ")
                return True

        if execute((action_buffers[1][7] + action_buffers[2][6]) / 2):
            print(colored("success", "green"), end=" ")
            _save_rollout_videos(img_dict, eval_dir, sequence_i, subtask_i, subtask, "succ")
            return True

        if execute(action_buffers[2][7]):
            print(colored("success", "green"), end=" ")
            _save_rollout_videos(img_dict, eval_dir, sequence_i, subtask_i, subtask, "succ")
            return True

    print(colored("fail", "red"), end=" ")
    _save_rollout_videos(img_dict, eval_dir, sequence_i, subtask_i, subtask, "fail")
    return False


def update_image_data(img_dict, obs):
    img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
    img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

def check_success(start_info, current_info, subtask, task_oracle):
    return len(task_oracle.get_task_info_for_set(start_info, current_info, {subtask})) > 0

def handle_success(img_dict, eval_dir, sequence_i, subtask_i, subtask, save_videos=False):
    print(colored("success", "green"), end=" ")
    _save_rollout_videos.enabled = bool(save_videos)
    _save_rollout_videos(img_dict, eval_dir, sequence_i, subtask_i, subtask, "succ")
    return True





from huggingface_hub import HfApi, hf_hub_download
import shutil
from datetime import datetime
import filecmp
from typing import Any, Dict, List, Optional, Tuple, Union
def model_is_on_hf_hub(model_path: str) -> bool:
    """Checks whether a model path points to a model on Hugging Face Hub."""
    # If the API call below runs without error, the model is on the hub
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False


def update_auto_map(pretrained_checkpoint: str) -> None:
    """
    Update the AutoMap configuration in the checkpoint config.json file.

    This loads the config.json file inside the checkpoint directory and overwrites
    the AutoConfig and AutoModelForVision2Seq fields to use OpenVLA-specific classes.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    raise RuntimeError(
        "Checkpoint mutation is disabled. Restore the official config.json "
        "and load checkpoint remote code read-only."
    )

    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: No config.json found at {config_path}")
        return

    # Create timestamped backup

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(pretrained_checkpoint, f"config.json.back.{timestamp}")
    shutil.copy2(config_path, backup_path)
    print(f"Created backup of original config at: {os.path.abspath(backup_path)}")

    # Read and update the config
    with open(config_path, "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }

    # Write back the updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated config.json at: {os.path.abspath(config_path)}")
    print("Changes made:")
    print('  - Set AutoConfig to "configuration_prismatic.OpenVLAConfig"')
    print('  - Set AutoModelForVision2Seq to "modeling_prismatic.OpenVLAForActionPrediction"')


def check_identical_files(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
    """
    Check if two files are identical in content.

    Args:
        path1: Path to the first file
        path2: Path to the second file

    Returns:
        bool: True if files are identical, False otherwise
    """
    path1, path2 = Path(path1), Path(path2)

    # First check if file sizes match
    if path1.stat().st_size != path2.stat().st_size:
        return False

    # Check if contents match
    return filecmp.cmp(path1, path2, shallow=False)




def _handle_file_sync(curr_filepath: str, checkpoint_filepath: str, file_type: str) -> None:
    """
    Handle syncing of files between current directory and checkpoint.

    Creates backups if files exist but differ, and copies current versions to checkpoint.

    Args:
        curr_filepath: Path to the current file version
        checkpoint_filepath: Path where the file should be in the checkpoint
        file_type: Description of the file type for logging
    """
    raise RuntimeError(
        "Checkpoint code synchronization is disabled during evaluation."
    )
    if os.path.exists(checkpoint_filepath):
        # Check if existing files are identical
        match = check_identical_files(curr_filepath, checkpoint_filepath)

        if not match:
            print(
                "\n------------------------------------------------------------------------------------------------\n"
                f"Found mismatch between:\n"
                f"Current:   {curr_filepath}\n"
                f"Checkpoint: {checkpoint_filepath}\n"
            )

            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{checkpoint_filepath}.back.{timestamp}"
            shutil.copy2(checkpoint_filepath, backup_path)
            print(f"Created backup of original checkpoint file at: {os.path.abspath(backup_path)}")

            # Copy current version to checkpoint directory
            shutil.copy2(curr_filepath, checkpoint_filepath)
            print(f"Copied current version to checkpoint at: {os.path.abspath(checkpoint_filepath)}")
            print(
                f"Changes complete. The checkpoint will now use the current version of {file_type}"
                "\n------------------------------------------------------------------------------------------------\n"
            )
    else:
        # If file doesn't exist in checkpoint directory, copy it
        shutil.copy2(curr_filepath, checkpoint_filepath)
        print(
            "\n------------------------------------------------------------------------------------------------\n"
            f"No {file_type} found in checkpoint directory.\n"
            f"Copied current version from: {curr_filepath}\n"
            f"To checkpoint location: {os.path.abspath(checkpoint_filepath)}"
            "\n------------------------------------------------------------------------------------------------\n"
        )


def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    """
    Check and sync model logic files between current code and checkpoint.

    Handles the relationship between current and checkpoint versions of both
    modeling_prismatic.py and configuration_prismatic.py:
    - If checkpoint file exists and differs: creates backup and copies current version
    - If checkpoint file doesn't exist: copies current version

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    raise RuntimeError(
        "Checkpoint code synchronization is disabled during evaluation."
    )

    # Find current files
    curr_files = {"modeling_prismatic.py": None, "configuration_prismatic.py": None}

    for root, _, files in os.walk("./prismatic/"):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)

    # Check and handle each file
    for filename, curr_filepath in curr_files.items():
        if curr_filepath is None:
            print(f"WARNING: `{filename}` is not found anywhere in the current directory.")
            continue

        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(curr_filepath, checkpoint_filepath, filename)


def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a component's state dict from checkpoint and handle DDP prefix if present.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict: The processed state dictionary for loading
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    # If the component was trained with DDP, elements in the state dict have prefix "module." which we must remove
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    """
    Find a specific checkpoint file matching a pattern.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
        file_pattern: String pattern to match in filenames

    Returns:
        str: Path to the matching checkpoint file

    Raises:
        AssertionError: If no files or multiple files match the pattern
    """
    assert os.path.isdir(pretrained_checkpoint), f"Checkpoint path must be a directory: {pretrained_checkpoint}"

    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)

    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in directory: {pretrained_checkpoint}"
    )

    return checkpoint_files[0]


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
import wandb
def setup_logging(cfg):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id



def _load_dataset_stats(vla: torch.nn.Module, checkpoint_path: str) -> None:
    """
    Load dataset statistics used during training for action normalization.

    Args:
        vla: The VLA model
        checkpoint_path: Path to the checkpoint directory
    """
    if model_is_on_hf_hub(checkpoint_path):
        # Download dataset stats directly from HF Hub
        dataset_statistics_path = hf_hub_download(
            repo_id=checkpoint_path,
            filename="dataset_statistics.json",
        )
    else:
        dataset_statistics_path = os.path.join(checkpoint_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )



MODEL_IMAGE_SIZES = {
    "openvla": 224,
    # Add other models as needed
}

def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    # if "image_aug" in str(cfg.pretrained_checkpoint):
    #     assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    if Path(str(cfg.pretrained_checkpoint)).name == "CALVIN-ABC-Pro":
        if not cfg.use_checkpoint_modeling:
            raise ValueError(
                "Official CALVIN-ABC-Pro must use --use_checkpoint_modeling True"
            )
        if not cfg.use_minivlm:
            raise ValueError("Official CALVIN-ABC-Pro requires --use_minivlm True")
        if not cfg.center_crop:
            raise ValueError("Official CALVIN-ABC-Pro requires --center_crop True")
        if not cfg.use_pro_version:
            raise ValueError("Official CALVIN-ABC-Pro requires --use_pro_version True")

    if cfg.validate_checkpoint_weights and os.path.isdir(cfg.pretrained_checkpoint):
        checkpoint_dir = Path(cfg.pretrained_checkpoint)
        model_weight_patterns = (
            "model.safetensors",
            "model-*.safetensors",
            "pytorch_model.bin",
            "pytorch_model-*.bin",
        )
        model_weights = [
            path
            for pattern in model_weight_patterns
            for path in checkpoint_dir.glob(pattern)
        ]
        action_heads = list(checkpoint_dir.glob("*action_head*checkpoint*.pt"))
        proprio_projectors = list(checkpoint_dir.glob("*proprio_projector*checkpoint*.pt"))
        missing = []
        if not model_weights:
            missing.append("merged VLA weights (model*.safetensors or pytorch_model*.bin)")
        if cfg.use_l1_regression and not action_heads:
            missing.append("action_head--*checkpoint*.pt")
        if cfg.use_proprio and not proprio_projectors:
            missing.append("proprio_projector--*checkpoint*.pt")
        if missing:
            raise FileNotFoundError(
                f"Incomplete checkpoint directory: {checkpoint_dir}. Missing: "
                + ", ".join(missing)
                + ". A tokenizer/config-only folder can load a model from the base "
                  "path or cache but cannot reproduce CALVIN-ABC-Pro. Copy the full "
                  "official checkpoint before evaluation."
            )

        if checkpoint_dir.name == "CALVIN-ABC-Pro":
            expected = {
                "config.json": (None, "05978420d096033d0fe7b4905f0178edfcc964f58c67f8776293b81e93d9a7b6"),
                "dataset_statistics.json": (None, "985bac37d251ee790d3e3f2d98c6102ace671817d3e532cf4b9a6b5e392cee82"),
                "configuration_prismatic.py": (6020, "87092aca7c0b91674bd837def049d9bf833beb1a99baf71a56a6c00628424dc4"),
                "modeling_prismatic.py": (69036, "f33a3bad5aff94072e33355fe7d7b5b559f4990f156e91fa966ef38d08fff754"),
                "processing_prismatic.py": (12718, "06563586bbc73d36501a14a231a3174904ede830d33174d0c625ece8e931ef1e"),
                "model.safetensors": (2505232584, "91c46e24b97bf25515af662ce314013e27fd7dfa56904b9e88090eb0e3bdbfa2"),
                "action_head--checkpoint.pt": (435954570, "dfb26818cd5e42243759a0b304d8616f26415b72976b6e2bb8549f1c9c0390f3"),
                "proprio_projector--checkpoint.pt": (1626096, "cb56b70db58dad041360818c4bd0a63a0f9f156048cbdb616a84f8e06e6608eb"),
            }
            problems = []
            for name, (expected_size, expected_sha) in expected.items():
                path = checkpoint_dir / name
                if not path.is_file():
                    problems.append(f"missing {name}")
                    continue
                if expected_size is not None and path.stat().st_size != expected_size:
                    problems.append(
                        f"{name} size={path.stat().st_size}, expected={expected_size}"
                    )
                # Full multi-GB hashes are opt-in because hashing the 2.5 GB VLM
                # weights can add substantial startup time on network storage.
                should_hash = expected_sha is not None and (
                    expected_size is None
                    or expected_size < 100_000_000
                    or os.environ.get("VLA_VALIDATE_FULL_SHA256", "0") == "1"
                )
                if should_hash:
                    hasher = hashlib.sha256()
                    with path.open("rb") as file:
                        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
                            hasher.update(block)
                    digest = hasher.hexdigest()
                    if digest != expected_sha:
                        problems.append(f"{name} sha256={digest}, expected={expected_sha}")
            if problems:
                raise RuntimeError(
                    "Official CALVIN-ABC-Pro checkpoint validation failed:\n  - "
                    + "\n  - ".join(problems)
                )
            print("[CALVIN] official checkpoint files validated (metadata hashes and weight sizes).")


def get_image_resize_size(model_family) -> Union[int, tuple]:
    return MODEL_IMAGE_SIZES[model_family]

def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)
    model.set_version(cfg.save_version)
    if cfg.use_checkpoint_modeling:
        model_module = model.__class__.__module__
        if not model_module.startswith("transformers_modules."):
            raise RuntimeError(
                "Expected Hugging Face dynamic checkpoint code, but loaded "
                f"{model_module}.{model.__class__.__name__}."
            )
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_spline_head or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        # check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def _resolve_path(path_value: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _validate_calvin_layout(cfg: GenerateConfig) -> Tuple[Path, Path]:
    """Validate the split CALVIN code/data layout before loading the model."""
    dataset_root = _resolve_path(cfg.calvin_path)
    dataset_path = dataset_root / "dataset" / "task_ABC_D"
    validation_path = dataset_path / "validation"
    if not validation_path.is_dir():
        raise FileNotFoundError(
            f"CALVIN dataset validation directory not found: {validation_path}. "
            "Expected --calvin_path/dataset/task_ABC_D/validation."
        )
    merged_config = validation_path / ".hydra" / "merged_config.yaml"
    if not merged_config.is_file():
        raise FileNotFoundError(
            f"CALVIN merged config not found: {merged_config}. "
            "Generate it with the config directory from the installed calvin_env checkout."
        )
    return dataset_path, validation_path

@draccus.wrap()
def main(cfg: GenerateConfig):
    seed_everything(cfg.seed)
    _print_runtime_sources()
    print(
        "[CALVIN] effective config: "
        f"checkpoint_modeling={cfg.use_checkpoint_modeling}, "
        f"minivlm={cfg.use_minivlm}, center_crop={cfg.center_crop}, "
        "rollout=hi3_temporal_ensemble, views=[rgb_static,rgb_gripper], "
        "unnorm_key=calvin_abc_rlds"
    )

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    acc = Accelerator(kwargs_handlers=[kwargs])
    # device = acc.device
    validate_config(cfg)
    dataset_path, _ = _validate_calvin_layout(cfg)
    if not CALVIN_ROOT:
        raise RuntimeError(
            "CALVIN_ROOT is not set. Export it to the CALVIN source checkout "
            "(for example /hy-tmp/calvin_repo) before running evaluation."
        )
    
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Keep CALVIN artifacts separate from LIBERO and from the legacy evaluator
    # output.  Each run gets a stable, self-contained directory containing the
    # metrics/result files; videos remain disabled by default.
    eval_root = _resolve_path(cfg.eval_log_dir)
    checkpoint_name = Path(str(cfg.pretrained_checkpoint).rstrip("/\\")).name or "checkpoint"
    eval_dir = eval_root / f"{current_time}_{checkpoint_name}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'],  # rgb_tactile
        # The CALVIN RGB smoke evaluator does not consume depth.  Requesting
        # depth streams here is incompatible with some CALVIN environment
        # versions and can result in an empty rgb_obs mapping.
        'depth_obs': [],
        'state_obs': ['robot_obs'],
        'actions': ['rel_actions'],
        'language': ['language']}
    env = make_env(str(dataset_path), observation_space, DEVICE)

    # Reset once before inspecting observations.  CALVIN versions that lazily
    # initialize cameras can return an empty rgb_obs from get_obs() before the
    # first reset.
    initial_obs = env.reset()
    if not (
        isinstance(initial_obs, dict)
        and isinstance(initial_obs.get('rgb_obs'), dict)
        and {'rgb_static', 'rgb_gripper'}.issubset(initial_obs['rgb_obs'])
    ):
        # A few CALVIN revisions return the observation from get_obs() rather
        # than from reset().  The wrapper caches the reset result and handles
        # this compatibility case.
        initial_obs = env.get_obs()
    rgb_obs = initial_obs.get('rgb_obs', {}) if isinstance(initial_obs, dict) else {}
    if not isinstance(rgb_obs, dict) or not {'rgb_static', 'rgb_gripper'}.issubset(rgb_obs):
        obs_keys = list(initial_obs.keys()) if isinstance(initial_obs, dict) else type(initial_obs)
        rgb_keys = list(rgb_obs.keys()) if isinstance(rgb_obs, dict) else type(rgb_obs)
        raise RuntimeError(
            'CALVIN did not provide the requested RGB observations. '
            f'obs_keys={obs_keys}, rgb_obs_keys={rgb_keys}. '
            'Check the calvin_env version and merged_config camera settings.'
        )


    runtime_metrics = CalvinRuntimeMetrics(warmup_queries=cfg.warmup_queries, action_chunk_size=8)
    parameter_metrics = parameter_summary(
        model,
        {
            "action_head": action_head,
            "proprio_projector": proprio_projector,
            "noisy_action_projector": noisy_action_projector,
        },
    )
    reset_cuda_peak_memory_stats()
    eva = DualSystemCalvinEvaluation(
        model,
        proprio_projector,
        noisy_action_projector,
        action_head,
        processor,
        use_x0_prediction=cfg.use_x0_prediction,
        runtime_metrics=runtime_metrics,
        use_minivlm=cfg.use_minivlm,
        center_crop=cfg.center_crop,
    )
    results, task_info = evaluate_policy(
        eva,
        env,
        str(eval_dir / 'success_rate.txt'),
        str(eval_dir / 'result.json'),
        acc.num_processes,
        acc.process_index,
        eval_dir=eval_dir,
        ep_len=360,
        num_sequences=cfg.num_sequences,
        enrich_lang=cfg.enrich_lang,
        debug=False,
        calvin_root=CALVIN_ROOT,
        dataset_root=str(cfg.calvin_path),
        runtime_metrics=runtime_metrics,
        save_videos=cfg.save_videos,
    )
    avg_reward = torch.tensor(results).float().mean().to(DEVICE)

    memory_metrics = get_cuda_memory_stats()
    metrics_payload = build_metrics_payload(
        results=results,
        runtime=runtime_metrics,
        memory=memory_metrics,
        parameters=parameter_metrics,
        task_info=task_info,
        metadata={
            "seed": cfg.seed,
            "checkpoint": str(cfg.pretrained_checkpoint),
            "calvin_path": str(cfg.calvin_path),
            "num_sequences": cfg.num_sequences,
            "warmup_queries": cfg.warmup_queries,
            "action_chunk_size": 8,
            "rollout": "official_hi3_temporal_ensemble",
            "checkpoint_model_class": (
                f"{model.__class__.__module__}.{model.__class__.__name__}"
            ),
            "processor_class": (
                f"{processor.__class__.__module__}.{processor.__class__.__name__}"
                if processor is not None
                else None
            ),
            "egl_enabled": bool(getattr(env.env, "use_egl", False)),
            "cuda_device": str(DEVICE),
        },
    )
    print_markdown_summary(metrics_payload)

    # Always emit the complete profiling payload.  ``--metrics_output`` is an
    # optional override; without it, keep the JSON beside the run's videos and
    # legacy result.txt file.
    metrics_path = _resolve_path(cfg.metrics_output) if cfg.metrics_output else eval_dir / "metrics.json"
    write_metrics_json(metrics_payload, metrics_path)
    logger.info("Wrote metrics to %s", metrics_path)

    acc.wait_for_everyone()
    avg_reward = acc.gather_for_metrics(avg_reward).mean()
    if acc.is_main_process:
        print('average success rate ', avg_reward)


if __name__ == "__main__":
    main()
