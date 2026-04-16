"""Shared pickle format for deferred W&B logging from resume_* scripts."""
import os
import pickle
from typing import Any, Dict, List, Optional

PENDING_VERSION = 1


def load_pending(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return _empty_doc()
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or data.get("version") != PENDING_VERSION:
        return _empty_doc()
    return data


def _empty_doc() -> Dict[str, Any]:
    return {
        "version": PENDING_VERSION,
        "wandb_project": None,
        "wandb_entity": None,
        "rm_metric": {},
        "grpo_finegrained_eval": {},
    }


def save_pending_merge(
    path: str,
    *,
    wandb_project: str,
    wandb_entity: Optional[str],
    rm_metric: Optional[Dict[str, Dict[str, Any]]] = None,
    grpo_finegrained_eval: Optional[Dict[str, Any]] = None,
) -> None:
    data = load_pending(path)
    data["wandb_project"] = wandb_project
    if wandb_entity is not None:
        data["wandb_entity"] = wandb_entity
    if rm_metric:
        data["rm_metric"].update(rm_metric)
    if grpo_finegrained_eval:
        data["grpo_finegrained_eval"].update(grpo_finegrained_eval)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def append_grpo_run_logs(path: str, run_id: str, log_sequence: List[Dict[str, Any]], **kwargs) -> None:
    save_pending_merge(path, grpo_finegrained_eval={run_id: {"log_sequence": log_sequence}}, **kwargs)


def append_rm_run(path: str, run_id: str, metrics: Dict[str, Any], **kwargs) -> None:
    save_pending_merge(path, rm_metric={run_id: metrics}, **kwargs)
