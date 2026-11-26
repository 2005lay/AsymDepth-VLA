"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import draccus
import numpy as np
import torch
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


BYTES_PER_GIB = 1024**3


@dataclass
class RuntimeMetrics:
    """End-to-end policy-query metrics for one returned action chunk."""

    model_query_times: List[float] = field(default_factory=list)
    actions_generated: int = 0
    episode_wall_times: List[float] = field(default_factory=list)
    vlm_times: List[float] = field(default_factory=list)
    policy_times: List[float] = field(default_factory=list)

    def add_model_query(self, elapsed_sec: float, num_actions: int) -> None:
        self.model_query_times.append(float(elapsed_sec))
        self.actions_generated += int(num_actions)

    def add_episode(self, elapsed_sec: float) -> None:
        self.episode_wall_times.append(float(elapsed_sec))


def synchronize_cuda_if_available() -> None:
    """Make the wall-clock query interval include queued CUDA work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_cuda_peak_memory_stats() -> None:
    if torch.cuda.is_available():
        synchronize_cuda_if_available()
        torch.cuda.reset_peak_memory_stats()


def get_cuda_memory_stats() -> Optional[Dict[str, Union[float, str]]]:
    if not torch.cuda.is_available():
        return None
    synchronize_cuda_if_available()
    device = torch.cuda.current_device()
    return {
        "device": torch.cuda.get_device_name(device),
        "allocated_gib": torch.cuda.memory_allocated(device) / BYTES_PER_GIB,
        "reserved_gib": torch.cuda.memory_reserved(device) / BYTES_PER_GIB,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / BYTES_PER_GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / BYTES_PER_GIB,
    }


def summarize_runtime_metrics(metrics: RuntimeMetrics) -> Dict[str, float]:
    query = np.asarray(metrics.model_query_times, dtype=np.float64)
    episodes = np.asarray(metrics.episode_wall_times, dtype=np.float64)
    result: Dict[str, float] = {
        "policy_queries": int(query.size),
        "actions_generated": int(metrics.actions_generated),
    }
    if query.size:
        result.update({
            "total_model_query_time_sec": float(query.sum()),
            "total_policy_query_time_sec": float(query.sum()),
            "avg_model_query_time_sec": float(query.mean()),
            "median_model_query_time_sec": float(np.median(query)),
            "p95_model_query_time_sec": float(np.percentile(query, 95)),
            "min_model_query_time_sec": float(query.min()),
            "max_model_query_time_sec": float(query.max()),
            "avg_actions_per_query": float(metrics.actions_generated / query.size),
            "avg_time_per_generated_action_sec": float(query.sum() / max(metrics.actions_generated, 1)),
            # Explicit aliases used by the efficiency tables/JSON readers.
            # This is the legacy paper metric: one complete policy query that
            # returns an 8-action chunk, not one environment step.
            "avg_total_latency_ms_per_chunk": float(query.mean() * 1000.0),
            "amortized_latency_ms_per_returned_action": float(
                query.sum() * 1000.0 / max(metrics.actions_generated, 1)
            ),
        })
    if episodes.size:
        result.update({
            "total_eval_wall_time_sec": float(episodes.sum()),
            "avg_episode_wall_time_sec": float(episodes.mean()),
        })
    # Aggregate the table-contract component timers produced by get_vla_action.
    if metrics.vlm_times:
        result["avg_vlm_latency_ms_per_chunk"] = float(np.mean(metrics.vlm_times) * 1000.0)
        result["avg_policy_latency_ms_per_chunk"] = float(np.mean(metrics.policy_times) * 1000.0)
    result["component_timed_queries"] = len(metrics.vlm_times)
    return result


def log_runtime_summary(
    metrics: RuntimeMetrics,
    model_loaded_memory: Optional[Dict[str, Union[float, str]]],
    final_memory: Optional[Dict[str, Union[float, str]]],
    log_file=None,
) -> Dict[str, float]:
    """Print the legacy table-compatible latency and memory metrics."""
    summary = summarize_runtime_metrics(metrics)
    log_message("Runtime metrics:", log_file)
    log_message(f"Policy queries: {summary['policy_queries']}", log_file)
    log_message(f"Actions generated: {summary['actions_generated']}", log_file)
    if "avg_vlm_latency_ms_per_chunk" in summary:
        log_message(f"VLM latency (ms/chunk): {summary['avg_vlm_latency_ms_per_chunk']:.2f}", log_file)
        log_message(f"Policy latency (ms/chunk): {summary['avg_policy_latency_ms_per_chunk']:.2f}", log_file)
        log_message(
            "Component timing: VLM = policy image/text preprocessing through language-model output; "
            "Policy = language-model output through bridge slicing/rearrangement, proprioception "
            "encoding and the complete action head to the normalized 8-action chunk.",
            log_file,
        )
    if summary["policy_queries"]:
        log_message(
            "Policy query latency (sec/chunk): "
            f"avg={summary['avg_model_query_time_sec']:.4f}, "
            f"median={summary['median_model_query_time_sec']:.4f}, "
            f"p95={summary['p95_model_query_time_sec']:.4f}, "
            f"min={summary['min_model_query_time_sec']:.4f}, "
            f"max={summary['max_model_query_time_sec']:.4f}",
            log_file,
        )
        # This is the value to copy to the paper table (multiply by 1000).
        log_message(
            f"Total latency (ms/chunk): {summary['avg_model_query_time_sec'] * 1000.0:.2f}",
            log_file,
        )
        log_message(
            f"Total end-to-end policy-query time: {summary['total_policy_query_time_sec']:.2f} sec",
            log_file,
        )
        log_message(f"Average actions per policy query: {summary['avg_actions_per_query']:.2f}", log_file)
        log_message(
            f"Average latency per generated action: {summary['avg_time_per_generated_action_sec']:.4f} sec",
            log_file,
        )
        log_message(
            "NOTE: table total latency = raw simulator observation to executable returned "
            "8-action chunk; environment step/render/video time is excluded.",
            log_file,
        )
    if "avg_episode_wall_time_sec" in summary:
        log_message(f"Average episode wall time: {summary['avg_episode_wall_time_sec']:.2f} sec", log_file)
        log_message(f"Total eval wall time: {summary['total_eval_wall_time_sec']:.2f} sec", log_file)
    if model_loaded_memory is not None:
        log_message(
            f"GPU memory after model load: allocated={model_loaded_memory['allocated_gib']:.2f} GiB, "
            f"reserved={model_loaded_memory['reserved_gib']:.2f} GiB",
            log_file,
        )
    if final_memory is not None:
        log_message(
            f"Peak GPU memory during eval: allocated={final_memory['peak_allocated_gib']:.2f} GiB, "
            f"reserved={final_memory['peak_reserved_gib']:.2f} GiB",
            log_file,
        )
    return summary



@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_minivlm: bool = True                         # If True, uses minivlm
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)
    require_component_timings: bool = False

    # fmt: on
    save_version: str = "vla-adapter"                # version of 
    use_pro_version: bool = True                     # encourage to use the pro models we released.
    phase: str = "Inference"



def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"



def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)
    model.set_version(cfg.save_version)
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
    if cfg.use_l1_regression:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


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



def setup_logging(cfg: GenerateConfig):
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



def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()



def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None



def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay



def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action



def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    runtime_metrics: Optional[RuntimeMetrics] = None,
    log_file=None,
):
    """Run a single episode in the environment."""
    episode_start_time = time.perf_counter()
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
               "{NUM_ACTIONS_CHUNK} constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Clear stale values outside the total-latency interval.
                model.last_vlm_time_sec = 0.0
                model.last_action_head_time_sec = 0.0
                # End-to-end policy-query latency starts from the raw simulator
                # observation and ends after the complete returned action chunk
                # has been converted into executable environment commands.  It
                # includes input extraction/resizing, processor CPU->GPU work,
                # VLM, bridge/action head, action de-normalization, gripper
                # post-processing, Python formatting/queueing, and CUDA fences.
                # Environment stepping, replay collection, rendering, and video
                # writing remain outside this interval.
                query_start_sec = time.perf_counter()
                synchronize_cuda_if_available()
                observation, img = prepare_observation(obs, resize_size)
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm
                )
                executable_actions = [
                    process_action(action, cfg.model_family).tolist() for action in actions
                ]
                action_queue.extend(executable_actions)
                synchronize_cuda_if_available()
                query_elapsed_sec = time.perf_counter() - query_start_sec
                if runtime_metrics is not None:
                    runtime_metrics.add_model_query(query_elapsed_sec, len(executable_actions))
                    vlm_sec = getattr(model, "last_vlm_time_sec", None)
                    policy_sec = getattr(model, "last_action_head_time_sec", None)
                    valid = (vlm_sec is not None and policy_sec is not None
                             and np.isfinite(vlm_sec) and np.isfinite(policy_sec)
                             and vlm_sec > 0 and policy_sec > 0)
                    if cfg.require_component_timings and not valid:
                        raise RuntimeError("Checkpoint lacks valid VLM/Policy timers; cannot report table metrics")
                    if valid:
                        runtime_metrics.vlm_times.append(float(vlm_sec))
                        runtime_metrics.policy_times.append(float(policy_sec))
            else:
                # Only the replay frame is needed while consuming a previously
                # generated action chunk; policy input preparation is performed
                # (and timed) exclusively when a new chunk is requested.
                img = get_libero_image(obs)

            # Replay bookkeeping is deliberately not part of policy latency.
            replay_images.append(img)

            # Get action from queue
            action = action_queue.popleft()

            # Execute action in environment
            obs, reward, done, info = env.step(action)
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)
        # A broken model/environment is not an unsuccessful task rollout.
        raise RuntimeError("Evaluation aborted on runtime error; do not use partial success rates") from e

    if runtime_metrics is not None:
        runtime_metrics.add_episode(time.perf_counter() - episode_start_time)

    return success, replay_images




def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    runtime_metrics: Optional[RuntimeMetrics] = None,
    log_file=None,
    save_version=None
):
    """Run evaluation for a single task."""
    # Get task
    # task_id = 8
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            runtime_metrics,
            log_file,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file, save_version=save_version
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)
    
    # close env
    env.close()
    del env

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes



@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Preserve the original experiment protocol: the official LIBERO default
    # initial-state set is deterministic and no evaluation seed is injected.
    # `cfg.seed` is retained only for backward-compatible config parsing.

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    model_loaded_memory = get_cuda_memory_stats()
    reset_cuda_peak_memory_stats()

    # for name, param in model.named_parameters():
    #     if 'action_queries' in name: 
    #         print(f"{name}: {param}")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    runtime_metrics = RuntimeMetrics()
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            runtime_metrics,
            log_file,
            cfg.save_version
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    runtime_summary = log_runtime_summary(
        runtime_metrics,
        model_loaded_memory,
        get_cuda_memory_stats(),
        log_file,
    )

    # Persist the same values in a machine-readable file.  In particular,
    # `avg_total_latency_ms_per_chunk` is the number to copy to the paper's
    # "Total latency / ms" column.  It spans raw simulator observation input
    # through an executable returned action chunk, while deliberately excluding
    # env.step(), rendering, video saving, and the whole-episode wall time.
    metrics_payload = {
        "checkpoint": str(cfg.pretrained_checkpoint),
        "task_suite": str(cfg.task_suite_name),
        "initial_states": str(cfg.initial_states_path),
        "seed_applied": False,
        "environment_seed": 0,
        "num_trials_per_task": cfg.num_trials_per_task,
        "num_open_loop_steps": cfg.num_open_loop_steps,
        "center_crop": cfg.center_crop,
        "vlm_kept_layers": getattr(model.config, "vlm_kept_layers", None),
        "policy_kept_layers": getattr(model.config, "policy_kept_layers", None),
        "feature_contract": getattr(model.config, "feature_contract", "official checkpoint"),
        "total_episodes": int(total_episodes),
        "total_successes": int(total_successes),
        "success_rate": float(final_success_rate),
        "policy_queries": int(runtime_summary.get("policy_queries", 0)),
        "actions_generated": int(runtime_summary.get("actions_generated", 0)),
        "latency_scope": "raw_observation_to_executable_action_chunk",
        "latency_excludes": ["env_step", "rendering", "video_writing", "episode_wall_time"],
        "avg_total_latency_ms_per_chunk": runtime_summary.get("avg_total_latency_ms_per_chunk"),
        "avg_vlm_latency_ms_per_chunk": runtime_summary.get("avg_vlm_latency_ms_per_chunk"),
        "avg_policy_latency_ms_per_chunk": runtime_summary.get("avg_policy_latency_ms_per_chunk"),
        "component_timed_queries": runtime_summary.get("component_timed_queries"),
        "component_latency_scope": {
            "vlm": "policy_image_text_preprocess_through_language_model_output",
            "policy": "language_model_output_through_bridge_preparation_and_action_head_normalized_chunk",
        },
        "amortized_latency_ms_per_returned_action": runtime_summary.get(
            "amortized_latency_ms_per_returned_action"
        ),
        "total_model_query_time_sec": runtime_summary.get("total_model_query_time_sec"),
        "total_policy_query_time_sec": runtime_summary.get("total_policy_query_time_sec"),
        "avg_episode_wall_time_sec": runtime_summary.get("avg_episode_wall_time_sec"),
        "total_eval_wall_time_sec": runtime_summary.get("total_eval_wall_time_sec"),
        "eval_memory": get_cuda_memory_stats(),
    }
    metrics_path = Path(cfg.local_log_dir) / f"{run_id}_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    log_message(f"Metrics JSON: {metrics_path}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate



if __name__ == "__main__":
    eval_libero()
