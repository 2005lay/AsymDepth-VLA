import os
import torch
import numpy as np
from PIL import Image
import time
from typing import Any, Dict
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
from calvin_metrics import CalvinRuntimeMetrics, synchronize_cuda
from experiments.robot.openvla_utils import center_crop_image, resize_image_for_policy


OPENVLA_IMAGE_SIZE = 224


def get_openvla_prompt(instruction: str, tokenized_action: str = None) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def get_qwen_chat_prompt(instruction: str) -> str:
    """Return the prompt used only by checkpoints trained with MiniVLM mode."""
    return (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        "<|im_end|>\n"
        f"<|im_start|>user\nWhat action should the robot take to {instruction.lower()}?"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio


def check_image_format(image: Any) -> None:
    """
    Validate input image format.

    Args:
        image: Image to check

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8

    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! Make sure that the input image is a "
        "numpy array with shape (H, W, 3) and dtype np.uint8!"
    )



class DualSystemCalvinEvaluation(CalvinBaseModel):
    def __init__(
        self,
        model,
        proprio_projector,
        noisy_action_projector,
        action_head,
        processor,
        use_x0_prediction=False,
        runtime_metrics=None,
        use_minivlm=False,
        center_crop=True,
    ):
        super().__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.processor = processor
        self.OFT = model
        self.proprio_projector = proprio_projector
        self.noisy_action_projector = noisy_action_projector
        self.action_head = action_head
        self.use_x0_prediction = use_x0_prediction
        self.runtime_metrics = runtime_metrics
        self.chain_position = None
        # Training uses PurePromptBuilder by default (In:/Out:).  The Qwen
        # chat template is only valid for runs explicitly trained with
        # use_minivlm=True.  Keep this switch explicit so evaluation cannot
        # silently change the language distribution seen by the checkpoint.
        self.use_minivlm = bool(use_minivlm)
        self.center_crop = bool(center_crop)
        self._prompt_logged = False
        self._pipeline_logged = False
        self._debug_action_count = 0

        # Set x0 prediction flag in action head if using diffusion
        if self.action_head is not None and hasattr(self.action_head, 'use_x0_prediction'):
            self.action_head.use_x0_prediction = use_x0_prediction

        self.temporal_size = 8
        self.temporal_mask = torch.flip(torch.triu(torch.ones(self.temporal_size, self.temporal_size, dtype=torch.bool)), dims=[1]).numpy()
        
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

        self.action = None
        self.hidden_states = None
        self.obs_buffer = None

        # Action chunking with temporal aggregation
        balancing_factor = 0.1
        self.temporal_weights = np.array([np.exp(-1 * balancing_factor * i) for i in range(self.temporal_size)])[:, None]

        # Dataset statics (rougnly computed with 10k samples in CALVIN)
        self.depth_max = 6.2
        self.depth_min = 3.5
        self.gripper_depth_max = 2.0
        self.gripper_depth_min = 0

        self.hist_action = []

        
    def reset(self,):
        """
        This is called
        """

        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)
        self.obs_buffer = None
        self.hist_action = []
        self._prompt_logged = False


    def step(self, obs, instruction, step):
        """
        Args:
            obs: environment observations
            instruction: embedded language goal
        Returns:
            action: predicted action
        """
        query_start = time.perf_counter()
        synchronize_cuda()
        image = obs["rgb_obs"]['rgb_static']  # {ndarray: (200, 200, 3)}
        gripper_image = obs["rgb_obs"]['rgb_gripper']  # {ndarray: (84, 84, 3)}
        # gripper_image1 = self.processor.image_processor.apply_transform(Image.fromarray(gripper_image))[:3].unsqueeze(0).to(self.dual_sys.device)

        # tactile_image = None
        # tactile_image = torch.from_numpy(obs["rgb_obs"]['rgb_tactile']).permute(2,0,1).unsqueeze(0).to(self.dual_sys.device, dtype=torch.float) / 255
        # depth_image = torch.from_numpy(obs["depth_obs"]['depth_static']).unsqueeze(0).to(self.dual_sys.device) - self.depth_min / (self.depth_max - self.depth_min)
        # depth_gripper = torch.from_numpy(obs["depth_obs"]['depth_gripper']).unsqueeze(0).to(self.dual_sys.device) - self.gripper_depth_min / (self.gripper_depth_max - self.gripper_depth_min)

        # Keep the camera order fixed as [static, gripper].  RLDS training
        # first resizes decoded frames to 224x224 and, when image_aug is on,
        # applies a square random-resized crop with scale=0.9.  Evaluation uses
        # the deterministic center-crop counterpart before the HF processor.
        check_image_format(image)
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)
        pil_image = Image.fromarray(np.ascontiguousarray(image), mode="RGB")

        check_image_format(gripper_image)
        if gripper_image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            gripper_image = resize_image_for_policy(
                gripper_image, OPENVLA_IMAGE_SIZE
            )
        gripper_pil_image = Image.fromarray(
            np.ascontiguousarray(gripper_image), mode="RGB"
        )
        if self.center_crop:
            pil_image = center_crop_image(pil_image)
            gripper_pil_image = center_crop_image(gripper_pil_image)
        if not self._pipeline_logged:
            print(
                "[CALVIN] visual pipeline: order=[rgb_static,rgb_gripper], "
                f"static={image.shape}/{image.dtype}, "
                f"gripper={gripper_image.shape}/{gripper_image.dtype}, "
                f"center_crop_0.9={self.center_crop}"
            )
            self._pipeline_logged = True
        prompt = (
            get_qwen_chat_prompt(instruction)
            if self.use_minivlm
            else get_openvla_prompt(instruction)
        )
        if not self._prompt_logged:
            print(
                "[CALVIN] prompt mode = "
                f"{'Qwen chat' if self.use_minivlm else 'PurePromptBuilder'}; "
                f"prompt={prompt!r}"
            )
            self._prompt_logged = True
        
        inputs = self.processor(prompt, pil_image).to(self.OFT.device, dtype=torch.bfloat16)
        wrist_inputs = self.processor(prompt, gripper_pil_image).to(
            self.OFT.device, dtype=torch.bfloat16
        )
        # inputs = self.processor(prompt, Image.fromarray(image)).to(self.OFT.device, dtype=torch.bfloat16)
        # all_wrist_inputs = [self.processor(prompt, Image.fromarray(gripper_image)).to(self.OFT.device, dtype=torch.bfloat16)]

        primary_pixel_values = inputs["pixel_values"]
        inputs["pixel_values"] = torch.cat(
            [primary_pixel_values, wrist_inputs["pixel_values"]], dim=1
        )

        # proprio_state = obs['robot_obs'][-8:]
        
        proprio_state = np.asarray(
            np.concatenate([obs['robot_obs'][:7], obs['robot_obs'][-1:]]),
            dtype=np.float32,
        )
        if proprio_state.shape != (8,):
            raise ValueError(
                f"Expected 8-D CALVIN proprio state, got {proprio_state.shape}"
            )
        proprio_norm_stats = self.OFT.norm_stats['calvin_abc_rlds']['proprio']
        # proprio_norm_stats = self.OFT.norm_stats['calvin']['proprio']

        obs["state"] = normalize_proprio(proprio_state, proprio_norm_stats)
        proprio_state = obs["state"]

        # state = torch.from_numpy(obs['robot_obs']).to(self.dual_sys.device, dtype=torch.float)
        # state = torch.cat([state[:6], state[[-1]]], dim=-1).unsqueeze(0)\
        with torch.no_grad():
            action, _ = self.OFT.predict_action(
                **inputs,
                unnorm_key="calvin_abc_rlds",
                # unnorm_key="calvin",
                do_sample=False,
                proprio=proprio_state,
                proprio_projector=self.proprio_projector,
                action_head=self.action_head,
                noisy_action_projector=self.noisy_action_projector,
                use_film=False,
            )
        debug_action_limit = int(os.environ.get("VLA_DEBUG_ACTION_LIMIT", "20"))
        if (
            os.environ.get("VLA_DEBUG_ACTIONS", "0") == "1"
            and self._debug_action_count < debug_action_limit
        ):
            action_np = np.asarray(action, dtype=np.float32)
            print(
                "[CALVIN][action-debug] "
                f"instruction={instruction!r} step={step} shape={action_np.shape} "
                f"range=({action_np.min():+.4f},{action_np.max():+.4f}) "
                f"first6_mean={action_np[:, :6].mean(axis=0)} "
                f"gripper={action_np[:, -1]}"
            )
            self._debug_action_count += 1
        synchronize_cuda()
        query_elapsed_ms = (time.perf_counter() - query_start) * 1000.0
        if self.runtime_metrics is not None:
            self.runtime_metrics.record_query(
                total_wall_ms=query_elapsed_ms,
                model_cuda_ms=getattr(self.OFT, "last_total_cuda_ms", None),
                vlm_cuda_ms=getattr(self.OFT, "last_vlm_cuda_ms", None),
                policy_cuda_ms=getattr(self.OFT, "last_policy_cuda_ms", None),
                actions_generated=min(len(action), self.runtime_metrics.action_chunk_size),
                chain_position=self.chain_position,
            )
        

        return [action[i] for i in range(min(len(action), 8))]
