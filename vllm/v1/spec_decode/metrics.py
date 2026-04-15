# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from dataclasses import dataclass, field

import numpy as np
import prometheus_client

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class SpecDecodingStats:
    """Per-step iteration decoding stats from scheduler.

    Each scheduler step, statistics on spec decoding performance are
    aggregated across requests by the scheduler and returned to the
    frontend in EngineCoreOutputs->SchedulerStats.
    """

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)
    # Top-k acceptance tracking (for analysis only).
    # num_topk_accepted_per_pos[k][pos]: how many times target token at pos
    # was in draft model's top-(k+1) predictions.
    num_topk_accepted_per_pos: list[list[int]] = field(default_factory=list)
    # Total evaluated positions (for normalization).
    num_topk_total_per_pos: list[int] = field(default_factory=list)

    @classmethod
    def new(cls, num_spec_tokens: int, topk: int = 3) -> "SpecDecodingStats":
        return cls(
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens_per_pos=[0] * num_spec_tokens,
            num_topk_accepted_per_pos=[
                [0] * num_spec_tokens for _ in range(topk)
            ],
            num_topk_total_per_pos=[0] * num_spec_tokens,
        )

    def observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int):
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        assert num_accepted_tokens <= self.num_spec_tokens
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1

    def observe_topk(
        self,
        topk_hits: list[list[int]],
        topk_total: list[int],
    ):
        """Observe top-k acceptance hits from a batch.

        Args:
            topk_hits: [topk][num_spec_tokens] hit counts per k per position.
            topk_total: [num_spec_tokens] total evaluated per position.
        """
        for k in range(len(topk_hits)):
            if k < len(self.num_topk_accepted_per_pos):
                for pos in range(
                    min(len(topk_hits[k]), len(self.num_topk_accepted_per_pos[k]))
                ):
                    self.num_topk_accepted_per_pos[k][pos] += topk_hits[k][pos]
        for pos in range(
            min(len(topk_total), len(self.num_topk_total_per_pos))
        ):
            self.num_topk_total_per_pos[pos] += topk_total[pos]


class SpecDecodingLogging:
    """Aggregate and log spec decoding metrics.

    LoggingStatLogger aggregates per-iteration metrics over a set
    time interval using observe() and then logs them using log()
    before resetting to zero.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.num_drafts: list[int] = []
        self.num_draft_tokens: list[int] = []
        self.num_accepted_tokens: list[int] = []
        self.accepted_tokens_per_pos_lists: list[list[int]] = []
        self.topk_accepted_lists: list[list[list[int]]] = []
        self.topk_total_lists: list[list[int]] = []
        self.last_log_time = time.monotonic()

    def observe(self, spec_decoding_stats: SpecDecodingStats):
        self.num_drafts.append(spec_decoding_stats.num_drafts)
        self.num_draft_tokens.append(spec_decoding_stats.num_draft_tokens)
        self.num_accepted_tokens.append(spec_decoding_stats.num_accepted_tokens)
        self.accepted_tokens_per_pos_lists.append(
            spec_decoding_stats.num_accepted_tokens_per_pos
        )
        self.topk_accepted_lists.append(
            spec_decoding_stats.num_topk_accepted_per_pos
        )
        self.topk_total_lists.append(
            spec_decoding_stats.num_topk_total_per_pos
        )

    def log(self, log_fn=logger.info):
        if not self.num_drafts:
            return
        num_drafts = np.sum(self.num_drafts)
        num_draft_tokens = np.sum(self.num_draft_tokens)
        num_accepted_tokens = np.sum(self.num_accepted_tokens)
        draft_throughput = 0
        accepted_throughput = 0

        elapsed_time = time.monotonic() - self.last_log_time
        if elapsed_time > 0:
            draft_throughput = num_draft_tokens / elapsed_time
            accepted_throughput = num_accepted_tokens / elapsed_time

        draft_acceptance_rate = (
            num_accepted_tokens / num_draft_tokens * 100
            if num_draft_tokens > 0
            else float("nan")
        )

        # Conventionally, mean acceptance length includes the bonus token
        mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)

        pos_matrix = np.array(self.accepted_tokens_per_pos_lists)
        acceptance_rates = np.sum(pos_matrix, axis=0) / num_drafts
        rates_str = ", ".join(f"{p:.3f}" for p in acceptance_rates)

        # Top-k acceptance rates
        topk_str = ""
        if self.topk_accepted_lists and self.topk_total_lists:
            topk_total = np.sum(
                np.array(self.topk_total_lists), axis=0
            )
            for k_idx, k_label in enumerate(["top1", "top2", "top3"]):
                if k_idx < len(self.topk_accepted_lists[0]):
                    k_hits = np.sum(
                        [lst[k_idx] for lst in self.topk_accepted_lists
                         if k_idx < len(lst)],
                        axis=0,
                    )
                    k_rates = np.where(
                        topk_total > 0,
                        k_hits / topk_total,
                        0.0,
                    )
                    k_rates_str = ", ".join(f"{r:.3f}" for r in k_rates)
                    topk_str += f", {k_label}: [{k_rates_str}]"

        log_fn(
            "SpecDecoding metrics: "
            "Mean acceptance length: %.2f, "
            "Accepted throughput: %.2f tokens/s, "
            "Drafted throughput: %.2f tokens/s, "
            "Accepted: %d tokens, "
            "Drafted: %d tokens, "
            "Per-position acceptance rate: %s, "
            "Avg Draft acceptance rate: %.1f%%%s",
            mean_acceptance_length,
            accepted_throughput,
            draft_throughput,
            num_accepted_tokens,
            num_draft_tokens,
            rates_str,
            draft_acceptance_rate,
            topk_str,
        )
        self.reset()


class SpecDecodingProm:
    """Record spec decoding metrics in Prometheus.

    The acceptance rate can be calculated using a PromQL query:

      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_draft_tokens_total[$interval])

    The mean acceptance length (conventionally including bonus tokens)
    can be calculated using:

      1 + (
      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_drafts[$interval]))

    A per-position acceptance rate vector can be computed using

      vllm:spec_decode_num_accepted_tokens_per_pos[$interval] /
      vllm:spec_decode_num_drafts[$interval]
    """

    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        speculative_config: SpeculativeConfig | None,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        self.spec_decoding_enabled = speculative_config is not None
        if not self.spec_decoding_enabled:
            return

        counter_drafts = self._counter_cls(
            name="vllm:spec_decode_num_drafts",
            documentation="Number of spec decoding drafts.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_drafts = make_per_engine(
            counter_drafts, per_engine_labelvalues
        )

        counter_draft_tokens = self._counter_cls(
            name="vllm:spec_decode_num_draft_tokens",
            documentation="Number of draft tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_draft_tokens = make_per_engine(
            counter_draft_tokens, per_engine_labelvalues
        )

        counter_accepted_tokens = self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens",
            documentation="Number of accepted tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens = make_per_engine(
            counter_accepted_tokens, per_engine_labelvalues
        )

        assert speculative_config is not None
        num_spec_tokens = (
            speculative_config.num_speculative_tokens
            if self.spec_decoding_enabled
            else 0
        )
        pos_labelnames = labelnames + ["position"]
        base_counter = self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens_per_pos",
            documentation="Accepted tokens per draft position.",
            labelnames=pos_labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens_per_pos: dict[
            int, list[prometheus_client.Counter]
        ] = {
            idx: [base_counter.labels(*lv, str(pos)) for pos in range(num_spec_tokens)]
            for idx, lv in per_engine_labelvalues.items()
        }

        # Top-k acceptance counters: labeled by (position, topk).
        topk_labelnames = labelnames + ["position", "topk"]
        topk_counter = self._counter_cls(
            name="vllm:spec_decode_num_topk_accepted_per_pos",
            documentation="Top-k accepted tokens per draft position.",
            labelnames=topk_labelnames,
        )
        self.counter_topk_accepted: dict[
            int, list[list[prometheus_client.Counter]]
        ] = {
            idx: [
                [topk_counter.labels(*lv, str(pos), str(k + 1))
                 for pos in range(num_spec_tokens)]
                for k in range(3)
            ]
            for idx, lv in per_engine_labelvalues.items()
        }

        topk_total_labelnames = labelnames + ["position"]
        topk_total_counter = self._counter_cls(
            name="vllm:spec_decode_num_topk_total_per_pos",
            documentation="Total evaluated drafts per position for top-k.",
            labelnames=topk_total_labelnames,
        )
        self.counter_topk_total: dict[
            int, list[prometheus_client.Counter]
        ] = {
            idx: [topk_total_counter.labels(*lv, str(pos))
                  for pos in range(num_spec_tokens)]
            for idx, lv in per_engine_labelvalues.items()
        }

    def observe(self, spec_decoding_stats: SpecDecodingStats, engine_idx: int = 0):
        if not self.spec_decoding_enabled:
            return
        self.counter_spec_decode_num_drafts[engine_idx].inc(
            spec_decoding_stats.num_drafts
        )
        self.counter_spec_decode_num_draft_tokens[engine_idx].inc(
            spec_decoding_stats.num_draft_tokens
        )
        self.counter_spec_decode_num_accepted_tokens[engine_idx].inc(
            spec_decoding_stats.num_accepted_tokens
        )
        for pos, counter in enumerate(
            self.counter_spec_decode_num_accepted_tokens_per_pos[engine_idx]
        ):
            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])

        # Top-k counters
        for k in range(len(spec_decoding_stats.num_topk_accepted_per_pos)):
            if k < len(self.counter_topk_accepted.get(engine_idx, [])):
                for pos, count in enumerate(
                    spec_decoding_stats.num_topk_accepted_per_pos[k]
                ):
                    self.counter_topk_accepted[engine_idx][k][pos].inc(count)
        for pos, count in enumerate(
            spec_decoding_stats.num_topk_total_per_pos
        ):
            if pos < len(self.counter_topk_total.get(engine_idx, [])):
                self.counter_topk_total[engine_idx][pos].inc(count)


def make_per_engine(
    counter: prometheus_client.Counter,
    per_engine_labelvalues: dict[int, list[object]],
):
    """Create a counter for each label value."""
    return {
        idx: counter.labels(*labelvalues)
        for idx, labelvalues in per_engine_labelvalues.items()
    }
