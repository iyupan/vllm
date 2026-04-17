# SPDX-License-Identifier: Apache-2.0
"""
Global accumulator for MTP top-k acceptance statistics.

Since ModelRunnerOutput dataclass fields are lost during cross-process
pickle transport in vLLM V1, we bypass the ModelRunnerOutput pipeline
and accumulate stats in-process.  The accumulated results are written
to a JSON file that the test script reads after generation completes.
"""

import atexit
import json
import os
import tempfile
import time
from pathlib import Path

import torch

# Module-level accumulators (per worker process).
_topk_hits_acc: list[list[int]] | None = None
_topk_total_acc: list[int] | None = None
_stats_path: str | None = None
# Identifier for the current accumulator run, embedded in the JSON so
# readers can detect stale files left by a previous run.
_run_id: str | None = None


def _make_run_id() -> str:
    return f"{os.getpid()}-{int(time.time() * 1000)}"


def _get_stats_path() -> str:
    """Return a deterministic path for the stats file."""
    global _stats_path
    if _stats_path is None:
        _stats_path = os.environ.get(
            "VLLM_TOPK_STATS_PATH",
            str(Path(tempfile.gettempdir()) / "vllm_topk_stats.json"),
        )
    return _stats_path


def _is_rank_zero() -> bool:
    """Check if this is TP rank 0 (safe to call even without distributed)."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def accumulate_topk_stats(
    topk_hits: torch.Tensor,
    topk_total: torch.Tensor,
) -> None:
    """Accumulate a batch of top-k hits/total into the global counters.

    Only TP rank 0 writes to disk to avoid concurrent file writes.

    Args:
        topk_hits: [topk, num_spec_steps] int tensor.
        topk_total: [num_spec_steps] int tensor.
    """
    if not _is_rank_zero():
        return

    global _topk_hits_acc, _topk_total_acc, _run_id

    hits = topk_hits.tolist()
    total = topk_total.tolist()

    if _topk_hits_acc is None:
        _topk_hits_acc = [[0] * len(hits[0]) for _ in range(len(hits))]
        _topk_total_acc = [0] * len(total)
        if _run_id is None:
            _run_id = _make_run_id()

    for k in range(len(hits)):
        for pos in range(len(hits[k])):
            _topk_hits_acc[k][pos] += hits[k][pos]
    for pos in range(len(total)):
        _topk_total_acc[pos] += total[pos]

    # Auto-flush to disk on every accumulation so the file is always
    # up-to-date (the data is tiny, so the overhead is negligible).
    flush_topk_stats()


def flush_topk_stats() -> None:
    """Write accumulated stats to disk (called at the end of generation)."""
    if _topk_hits_acc is None:
        return
    path = _get_stats_path()
    data = {
        "topk_hits": _topk_hits_acc,
        "topk_total": _topk_total_acc,
        "run_id": _run_id,
    }
    with open(path, "w") as f:
        json.dump(data, f)


def reset_topk_stats(remove_file: bool = True) -> None:
    """Reset the global accumulators.

    Also removes the on-disk stats file by default so a subsequent reader
    cannot see stats from a prior run merged with (or mistaken for) the
    current one. Pass ``remove_file=False`` to keep the file — useful when
    tests want to inspect state without a side effect.
    """
    global _topk_hits_acc, _topk_total_acc, _run_id
    _topk_hits_acc = None
    _topk_total_acc = None
    _run_id = None
    if remove_file:
        try:
            os.remove(_get_stats_path())
        except FileNotFoundError:
            pass


def load_topk_stats(
    path: str | None = None,
    expected_run_id: str | None = None,
) -> dict | None:
    """Load accumulated stats from disk (called by the test script).

    If ``expected_run_id`` is provided and does not match the file's
    ``run_id``, returns ``None`` — the file is stale (from a prior run).
    The returned dict includes the ``run_id`` field so callers can report
    it alongside their metrics.
    """
    path = path or _get_stats_path()
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if (expected_run_id is not None
            and data.get("run_id") != expected_run_id):
        return None
    return data


# Flush on process exit as a safety net.
atexit.register(flush_topk_stats)