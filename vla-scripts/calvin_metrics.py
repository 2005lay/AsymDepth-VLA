"""Runtime metrics for CALVIN evaluation.

The metric definitions intentionally mirror the profiling fields used by the
LIBERO evaluator:

* latency is measured per policy query (an 8-action chunk),
* the first ``warmup_queries`` are excluded from latency summaries,
* peak memory is ``max_memory_allocated`` after a pre-evaluation reset, and
* model parameters are reported as VLM versus policy components.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch


BYTES_PER_GIB = 1024 ** 3


def synchronize_cuda() -> None:
    """Synchronize CUDA before reading wall-clock or memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_cuda_peak_memory_stats() -> None:
    """Reset the allocated-memory peak immediately before evaluation."""
    if torch.cuda.is_available():
        synchronize_cuda()
        torch.cuda.reset_peak_memory_stats()


def get_cuda_memory_stats() -> Optional[Dict[str, Any]]:
    """Return allocated/reserved/peak CUDA memory in GiB."""
    if not torch.cuda.is_available():
        return None

    synchronize_cuda()
    device = torch.cuda.current_device()
    stats: Dict[str, Any] = {
        "device": torch.cuda.get_device_name(device),
        "allocated_gib": torch.cuda.memory_allocated(device) / BYTES_PER_GIB,
        "reserved_gib": torch.cuda.memory_reserved(device) / BYTES_PER_GIB,
        "max_allocated_gib": torch.cuda.max_memory_allocated(device) / BYTES_PER_GIB,
        "max_reserved_gib": torch.cuda.max_memory_reserved(device) / BYTES_PER_GIB,
    }
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        stats["free_gib"] = free_bytes / BYTES_PER_GIB
        stats["total_gib"] = total_bytes / BYTES_PER_GIB
    except RuntimeError:
        pass
    return stats


def count_parameters(module: Optional[torch.nn.Module]) -> int:
    """Count physical parameters, including frozen parameters."""
    if module is None:
        return 0
    return int(sum(parameter.numel() for parameter in module.parameters()))


def parameter_summary(
    vlm: torch.nn.Module,
    policy_modules: Mapping[str, Optional[torch.nn.Module]],
) -> Dict[str, Any]:
    """Split physical parameters into the paper's VLM/policy components.

    ``OpenVLAForActionPrediction`` stores the learned action-query tokens on
    the VLM object itself, although they are the bridge input consumed by the
    policy.  Move that small bridge component to the policy side so the two
    columns do not depend on the Python module nesting.
    """
    bridge_module = getattr(vlm, "action_queries", None)
    bridge_params = count_parameters(bridge_module)
    vlm_params = max(count_parameters(vlm) - bridge_params, 0)
    policy_components = {
        name: count_parameters(module) for name, module in policy_modules.items() if module is not None
    }
    if bridge_module is not None:
        policy_components["action_queries_bridge"] = bridge_params
    policy_params = int(sum(policy_components.values()))
    total_params = vlm_params + policy_params
    return {
        "vlm_params": vlm_params,
        "policy_params": policy_params,
        "total_params": total_params,
        "vlm_params_b": vlm_params / 1e9,
        "policy_params_m": policy_params / 1e6,
        "total_params_b": total_params / 1e9,
        "policy_components": policy_components,
        "definition": (
            "VLM is vision_backbone + multimodal projector + language_model; "
            "Policy is the action-query bridge plus action head and "
            "proprio/noisy-action projectors."
        ),
    }


@dataclass
class CalvinRuntimeMetrics:
    """Collect per-query latency with a fixed warmup policy."""

    warmup_queries: int = 10
    action_chunk_size: int = 8
    query_count: int = 0
    records: List[Dict[str, Any]] = field(default_factory=list)

    def record_query(
        self,
        *,
        total_wall_ms: float,
        model_cuda_ms: Optional[float],
        vlm_cuda_ms: Optional[float],
        policy_cuda_ms: Optional[float],
        actions_generated: int,
        chain_position: Optional[int],
    ) -> None:
        self.query_count += 1
        record = {
            "query_index": self.query_count,
            "warmup": self.query_count <= self.warmup_queries,
            "chain_position": chain_position,
            "total_latency_ms": float(total_wall_ms),
            "model_cuda_latency_ms": None if model_cuda_ms is None else float(model_cuda_ms),
            "vlm_latency_ms": None if vlm_cuda_ms is None else float(vlm_cuda_ms),
            "policy_latency_ms": None if policy_cuda_ms is None else float(policy_cuda_ms),
            "actions_generated": int(actions_generated),
        }
        if vlm_cuda_ms is not None and policy_cuda_ms is not None:
            # This is the measured pipeline overhead (image/state processing,
            # transfers and action post-processing) left after subtracting the
            # two CUDA-event intervals from the end-to-end wall time.
            record["system_overhead_latency_ms"] = max(
                float(total_wall_ms) - float(vlm_cuda_ms) - float(policy_cuda_ms),
                0.0,
            )
        else:
            record["system_overhead_latency_ms"] = None
        # Keep warmup rows in the JSON detail log for auditability.  Summary
        # statistics select only rows with warmup == False below.
        self.records.append(record)

    @staticmethod
    def _summary(values: List[float]) -> Dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "mean_ms": float(array.mean()) if array.size else 0.0,
            "median_ms": float(np.median(array)) if array.size else 0.0,
            "p95_ms": float(np.percentile(array, 95)) if array.size else 0.0,
            "min_ms": float(array.min()) if array.size else 0.0,
            "max_ms": float(array.max()) if array.size else 0.0,
        }

    def _summarize_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        def values(key: str) -> List[float]:
            return [float(row[key]) for row in records if row.get(key) is not None]

        total = self._summary(values("total_latency_ms"))
        model_cuda = self._summary(values("model_cuda_latency_ms"))
        vlm = self._summary(values("vlm_latency_ms"))
        policy = self._summary(values("policy_latency_ms"))
        overhead = self._summary(values("system_overhead_latency_ms"))
        action_count = sum(int(row["actions_generated"]) for row in records)
        total_ms = float(sum(values("total_latency_ms")))
        # The paper protocol defines this as query latency divided by the
        # fixed generated chunk size (8), rather than by a variable number of
        # actions that happened to be returned by an individual query.
        average_action_ms = total_ms / max(len(records) * self.action_chunk_size, 1)
        return {
            "queries": len(records),
            "total_latency": total,
            "model_cuda_latency": model_cuda,
            "vlm_latency": vlm,
            "policy_latency": policy,
            "system_overhead_latency": overhead,
            "average_latency_per_generated_action_ms": average_action_ms,
            "actions_generated": action_count,
            "action_chunk_size": self.action_chunk_size,
        }

    def _measured_records(self) -> List[Dict[str, Any]]:
        return [row for row in self.records if not row.get("warmup", False)]

    def summary(self) -> Dict[str, Any]:
        measured_records = self._measured_records()
        by_chain_position: Dict[str, Any] = {}
        for position in range(1, 6):
            position_records = [
                row for row in measured_records if row.get("chain_position") == position
            ]
            by_chain_position[str(position)] = self._summarize_records(position_records)
        return {
            "warmup_queries": self.warmup_queries,
            "total_queries_seen": self.query_count,
            "measured_queries": len(measured_records),
            "all_queries": self._summarize_records(measured_records),
            "by_chain_position": by_chain_position,
            "records": self.records,
        }


def chain_performance_summary(results: List[int]) -> Dict[str, Any]:
    """Summarize CALVIN's 1--5 chain success rates and average length."""
    if not results:
        return {
            "num_sequences": 0,
            "chain_success_rate_percent": {str(i): 0.0 for i in range(1, 6)},
            "average_sequence_length": 0.0,
        }
    array = np.asarray(results, dtype=np.int64)
    return {
        "num_sequences": int(array.size),
        "chain_success_rate_percent": {
            str(i): float(np.mean(array >= i) * 100.0) for i in range(1, 6)
        },
        "average_sequence_length": float(array.mean()),
    }


def build_metrics_payload(
    *,
    results: List[int],
    runtime: CalvinRuntimeMetrics,
    memory: Optional[Dict[str, Any]],
    parameters: Dict[str, Any],
    metadata: Dict[str, Any],
    task_info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    performance = chain_performance_summary(results)
    return {
        "metric_definition": {
            "success_rate": "fraction of sequences completing at least N subtasks, reported as percent",
            "average_length": "mean number of consecutively completed subtasks, range 0 to 5",
            "latency": (
                "per 8-action policy query; first warmup queries excluded; VLM and "
                "action-head values are CUDA-event intervals, total is synchronized "
                "end-to-end wall time, and average action latency is total divided "
                "by fixed chunk size 8"
            ),
            "peak_memory": "torch.cuda.max_memory_allocated after reset_peak_memory_stats / 1024^3, reported as GB",
        },
        "performance": performance,
        # Keep the legacy name so the detailed per-task success counts remain
        # directly compatible with CALVIN's original result.txt schema.
        "task_info": dict(task_info or {}),
        "runtime": runtime.summary(),
        "memory": memory,
        "parameters": parameters,
        "metadata": metadata,
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def print_markdown_summary(payload: Dict[str, Any]) -> None:
    """Print table-2, table-3, and table-4 compatible summaries."""
    performance = payload["performance"]
    runtime = payload["runtime"]
    memory = payload.get("memory") or {}
    parameters = payload["parameters"]

    def mean_or_na(row: Dict[str, Any], field: str) -> str:
        summary = row[field]
        return _fmt(summary["mean_ms"]) if summary.get("count", 0) else "N/A"

    def action_mean_or_na(row: Dict[str, Any]) -> str:
        return (
            _fmt(row["average_latency_per_generated_action_ms"])
            if row.get("queries", 0)
            else "N/A"
        )

    def pct(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.2f}%"

    print("\n## CALVIN Table 2: Task Performance")
    print("| 1 | 2 | 3 | 4 | 5 | Avg. |")
    print("|---:|---:|---:|---:|---:|---:|")
    rates = performance["chain_success_rate_percent"]
    print(
        "| " + " | ".join(pct(rates.get(str(i))) for i in range(1, 6))
        + " | " + _fmt(performance["average_sequence_length"], 2) + " |"
    )

    print("\n## CALVIN Table 3: Time Efficiency (ms)")
    print("| Chain position | Action head | Avg. per generated action | VLM backbone | Total query |")
    print("|---:|---:|---:|---:|---:|")
    for position in range(1, 6):
        row = runtime["by_chain_position"][str(position)]
        print(
            f"| {position} | {mean_or_na(row, 'policy_latency')} | "
            f"{action_mean_or_na(row)} | "
            f"{mean_or_na(row, 'vlm_latency')} | "
            f"{mean_or_na(row, 'total_latency')} |"
        )

    print("\n## CALVIN Table 4: Space Efficiency (whole evaluation run)")
    print("| Scope | Peak allocated memory (GB) | VLM params (B) | Policy params (M) | Total params (B) |")
    print("|---|---:|---:|---:|---:|")
    print(
        f"| all | {_fmt(memory.get('max_allocated_gib'))} | "
        f"{_fmt(parameters['vlm_params_b'])} | "
        f"{_fmt(parameters['policy_params_m'])} | "
        f"{_fmt(parameters['total_params_b'])} |"
    )


def write_metrics_json(payload: Dict[str, Any], path: str | Path) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
