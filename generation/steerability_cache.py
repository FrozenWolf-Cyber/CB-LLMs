"""
Disk cache for steerability generations: one UTF-8 text file per (seed, concept, sample index).

Layout under the checkpoint folder (same directory as epoch weights):

    steerability_outputs/epoch_{E}[_lowscore]/
        c{idx:03d}_{sanitized_concept_name}/
            seed_{seed}_sample_{k}.txt

Used by resume steerability scripts and training eval so partial runs can resume.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence


def sanitize_concept_slug(name: str, max_len: int = 80) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name).strip())
    s = s.strip("_") or "concept"
    return s[:max_len]


def steerability_output_root(ckpt_prefix: str, epoch: int, is_low_score: bool) -> str:
    """Directory root for steerability samples for one evaluated checkpoint."""
    ckpt_prefix = os.path.normpath(ckpt_prefix)
    sfx = "_lowscore" if is_low_score else ""
    return os.path.join(ckpt_prefix, "steerability_outputs", f"epoch_{epoch}{sfx}")


def concept_subdir(root: str, concept_idx: int, concept_name: str) -> str:
    slug = sanitize_concept_slug(concept_name)
    return os.path.join(root, f"c{concept_idx:03d}_{slug}")


def sample_file_path(root: str, concept_idx: int, concept_name: str, seed: int, sample_idx: int) -> str:
    sub = concept_subdir(root, concept_idx, concept_name)
    return os.path.join(sub, f"seed_{seed}_sample_{sample_idx}.txt")


def read_sample(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_sample(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_concept_samples(
    cache_root: Optional[str],
    seed: int,
    concept_idx: int,
    concept_name: str,
    n_samples: int,
) -> List[Optional[str]]:
    """Return length-n list; entry k is file text or None if missing."""
    if not cache_root or n_samples <= 0:
        return [None] * max(0, n_samples)
    out: List[Optional[str]] = []
    for k in range(n_samples):
        p = sample_file_path(cache_root, concept_idx, concept_name, seed, k)
        out.append(read_sample(p))
    return out


def save_all_steerability_texts(
    cache_root: str,
    seed: int,
    concept_set: Sequence[str],
    texts_by_concept: Sequence[Sequence[str]],
) -> None:
    """
    Write every sample to disk (idempotent). Skips concepts with empty lists.
    Call at end of training / eval to guarantee on-disk snapshot.
    """
    if not cache_root:
        return
    for ci, texts in enumerate(texts_by_concept):
        if not texts or ci >= len(concept_set):
            continue
        cname = concept_set[ci]
        for si, t in enumerate(texts):
            p = sample_file_path(cache_root, ci, cname, seed, si)
            write_sample(p, t)
