"""
Upload metrics saved by resume_rm_metric_test.py / resume_grpo_finegrained_eval.py
when run with --skip_wandb --pending_wandb_pickle <path>.

Run this on a CPU-only or separate machine so GPU jobs are not blocked by W&B sync.
"""
import argparse
import os
import pickle

import wandb

from wandb_pending_io import PENDING_VERSION


def main():
    parser = argparse.ArgumentParser(
        description="Resume W&B runs and log metrics from a pending pickle (no model inference)."
    )
    parser.add_argument(
        "--pending_pickle",
        type=str,
        required=True,
        help="Path written by --pending_wandb_pickle from the resume scripts.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Override project (default: value stored in the pickle).",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Override entity (default: value stored in the pickle).",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=("all", "rm_metric", "grpo_finegrained_eval"),
        default="all",
        help="Which section to upload.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be logged without calling W&B.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.pending_pickle):
        raise SystemExit(f"Not found: {args.pending_pickle}")
    with open(args.pending_pickle, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or data.get("version") != PENDING_VERSION:
        raise SystemExit(
            f"Invalid pending pickle (expected version {PENDING_VERSION}): {args.pending_pickle}"
        )

    project = args.wandb_project or data.get("wandb_project")
    entity = args.wandb_entity if args.wandb_entity is not None else data.get("wandb_entity")
    if not project:
        raise SystemExit("wandb_project is not set in the pickle and --wandb_project was not passed.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def publish_rm():
        section = data.get("rm_metric") or {}
        for run_id, metrics in section.items():
            if args.dry_run:
                print(f"[dry_run] rm_metric {run_id}: {len(metrics)} keys")
                continue
            wandb.init(
                project=project,
                entity=entity,
                id=run_id,
                resume="must",
                save_code=False,
                settings=wandb.Settings(
                    console="off",
                    disable_git=True,
                    _disable_stats=True,
                ),
            )
            wandb.log(metrics)
            wandb.finish()

    def publish_grpo():
        section = data.get("grpo_finegrained_eval") or {}
        for run_id, payload in section.items():
            seq = payload.get("log_sequence") or []
            if args.dry_run:
                print(f"[dry_run] grpo_finegrained_eval {run_id}: {len(seq)} log calls")
                continue
            wandb.init(
                project=project,
                entity=entity,
                id=run_id,
                resume="must",
                save_code=False,
                settings=wandb.Settings(
                    console="off",
                    disable_git=True,
                    _disable_stats=True,
                ),
            )
            for chunk in seq:
                wandb.log(chunk)
            wandb.finish()

    if args.only in ("all", "rm_metric"):
        publish_rm()
    if args.only in ("all", "grpo_finegrained_eval"):
        publish_grpo()

    if args.dry_run:
        print("Dry run complete.")
    else:
        print("Done uploading pending metrics.")


if __name__ == "__main__":
    main()
