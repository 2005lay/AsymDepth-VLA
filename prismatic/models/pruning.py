"""Utilities for bridge-preserving layer pruning and checkpoint slicing.

Layer numbers exposed by the command line are one-based (the convention used in
the paper).  PyTorch ``ModuleList`` indices and Hugging Face checkpoint keys are
zero-based, so this module keeps that conversion in one place.
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import torch


DEFAULT_24_LAYERS = ",".join(str(i) for i in range(1, 25))


def parse_kept_layers(value: str | Sequence[int] | None, *, total_layers: int = 24) -> Tuple[int, ...]:
    """Parse and validate a one-based comma-separated layer list."""
    if value is None or value == "":
        value = DEFAULT_24_LAYERS
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        try:
            layers = tuple(int(item) for item in items)
        except ValueError as exc:
            raise ValueError(f"Invalid layer list {value!r}; expected e.g. '1,3,5'") from exc
    else:
        layers = tuple(int(item) for item in value)
    if not layers:
        raise ValueError("At least one layer must be kept")
    if len(set(layers)) != len(layers):
        raise ValueError(f"Layer list contains duplicates: {layers}")
    if tuple(sorted(layers)) != layers:
        raise ValueError(f"Layers must be in increasing original-depth order: {layers}")
    if any(layer < 1 or layer > total_layers for layer in layers):
        raise ValueError(f"Layer numbers must be in [1, {total_layers}], got {layers}")
    return layers


def layer_mapping(kept_layers: Sequence[int]) -> Dict[int, int]:
    """Return compact zero-based index -> full-model zero-based index mapping."""
    return {compact_idx: int(full_layer) - 1 for compact_idx, full_layer in enumerate(kept_layers)}


_LAYER_KEY_RE = re.compile(r"^(?P<prefix>.*\.(?:layers|mlp_resnet_blocks))\.(?P<idx>\d+)(?P<suffix>\..*)$")


def slice_state_dict(
    source_state_dict: Mapping[str, torch.Tensor],
    target_state_dict: Mapping[str, torch.Tensor],
    kept_layers: Sequence[int],
) -> Dict[str, torch.Tensor]:
    """Select full-model layer tensors and remap them to compact target keys.

    Non-layer tensors are copied by exact key match.  A tensor is only returned
    when its shape matches the compact target tensor; this makes the loader safe
    for checkpoints that contain optional LoRA or auxiliary entries.
    """
    mapping = layer_mapping(kept_layers)
    source = dict(source_state_dict)
    sliced: Dict[str, torch.Tensor] = {}

    for target_key, target_tensor in target_state_dict.items():
        source_key = target_key
        match = _LAYER_KEY_RE.match(target_key)
        if match:
            compact_idx = int(match.group("idx"))
            if compact_idx not in mapping:
                continue
            source_key = f"{match.group('prefix')}.{mapping[compact_idx]}{match.group('suffix')}"
        elif ".blocks." in target_key:
            # TIMM vision blocks are not part of the 24-layer language/policy
            # pruning contract; keep them by exact key when present.
            source_key = target_key
        if source_key in source and tuple(source[source_key].shape) == tuple(target_tensor.shape):
            sliced[target_key] = source[source_key]

    return sliced


def load_pruned_state_dict(
    module: torch.nn.Module,
    source_state_dict: Mapping[str, torch.Tensor],
    kept_layers: Sequence[int] | str,
    *,
    total_layers: int = 24,
    strict: bool = False,
) -> Tuple[Sequence[str], Sequence[str]]:
    """Load a sliced state dict into ``module`` and return missing/unexpected keys."""
    kept = parse_kept_layers(kept_layers, total_layers=total_layers)
    target = module.state_dict()
    sliced = slice_state_dict(source_state_dict, target, kept)
    result = module.load_state_dict(sliced, strict=False)
    if strict and (result.missing_keys or result.unexpected_keys):
        raise RuntimeError(f"Missing keys after pruned load: {result.missing_keys[:20]}")
    return result.missing_keys, result.unexpected_keys


def load_checkpoint_state_dict(checkpoint: str | Path) -> Dict[str, torch.Tensor]:
    """Load common HF/Prismatic checkpoint formats on CPU."""
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index_path = checkpoint / index_name
            if index_path.exists():
                index = json.loads(index_path.read_text(encoding="utf-8"))
                merged = {}
                for name in sorted(set(index["weight_map"].values())):
                    shard = (checkpoint / name).resolve()
                    if shard.parent != checkpoint.resolve():
                        raise ValueError(f"Invalid checkpoint shard path: {name}")
                    merged.update(load_checkpoint_state_dict(shard))
                missing = set(index["weight_map"]) - set(merged)
                if missing:
                    raise RuntimeError(f"Incomplete sharded checkpoint: {sorted(missing)[:10]}")
                return merged
        safetensors_path = checkpoint / "model.safetensors"
        pytorch_path = checkpoint / "pytorch_model.bin"
        candidates = [safetensors_path, pytorch_path]
        # Never mistake action_head/proprio/optimizer files for the VLM.
        for candidate in candidates:
            if candidate.exists():
                checkpoint = candidate
                break
        else:
            raise FileNotFoundError(f"No model checkpoint found under {checkpoint}")

    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(checkpoint), device="cpu")
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload in {checkpoint}")
    return {str(k).removeprefix("module."): v for k, v in state.items() if torch.is_tensor(v)}


def load_complete_state_dict(module, state, context="model"):
    """Strict loading, allowing only aliases proven to share a parameter."""
    state = dict(state)
    aliases = {}
    for name, param in module.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(param), []).append(name)
    for names in aliases.values():
        present = [name for name in names if name in state]
        if present:
            if any(not torch.equal(state[name], state[present[0]]) for name in present[1:]):
                raise RuntimeError(f"{context}: conflicting tied parameters {present}")
            for name in names:
                state.setdefault(name, state[present[0]])
    try:
        return module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"{context}: incomplete weights; random fallback is forbidden\n{exc}") from exc
