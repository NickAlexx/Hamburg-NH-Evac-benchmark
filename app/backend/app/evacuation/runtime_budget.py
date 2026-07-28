"""Shared runtime-budget semantics for the evacuation solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Callable, Optional


STRICT_BUDGET_MODE = "strict"
LEGACY_RESULTS_BUDGET_MODE = "legacy_results"
VALID_BUDGET_MODES = {STRICT_BUDGET_MODE, LEGACY_RESULTS_BUDGET_MODE}


def normalize_budget_mode(value: str) -> str:
    mode = str(value or STRICT_BUDGET_MODE).strip().lower()
    if mode not in VALID_BUDGET_MODES:
        choices = ", ".join(sorted(VALID_BUDGET_MODES))
        raise ValueError(f"Unknown budget_mode {value!r}; expected one of: {choices}.")
    return mode


@dataclass
class RuntimeBudget:
    """Track either the published search-loop budget or a strict run budget.

    ``legacy_results`` reproduces the boundary-based timing used to generate
    the stored published results: its clock starts when optimization begins.
    ``strict`` starts at solver entry and reserves a small amount of time for
    result construction.
    """

    limit_seconds: Optional[float]
    mode: str = STRICT_BUDGET_MODE
    postprocess_reserve_seconds: float = 0.25
    clock: Callable[[], float] = time.monotonic
    run_started_at: Optional[float] = None
    search_started_at: Optional[float] = field(default=None, init=False)
    deadline: Optional[float] = field(default=None, init=False)
    effective_reserve_seconds: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.mode = normalize_budget_mode(self.mode)
        self.run_started_at = (
            float(self.run_started_at)
            if self.run_started_at is not None
            else float(self.clock())
        )

        if self.limit_seconds is None:
            return

        self.limit_seconds = float(self.limit_seconds)
        if not math.isfinite(self.limit_seconds) or self.limit_seconds <= 0:
            self.limit_seconds = None
            return

        if self.mode == STRICT_BUDGET_MODE:
            requested = max(0.0, float(self.postprocess_reserve_seconds))
            self.effective_reserve_seconds = min(
                requested,
                self.limit_seconds * 0.10,
            )
            self.deadline = (
                self.run_started_at
                + self.limit_seconds
                - self.effective_reserve_seconds
            )

    @property
    def is_strict(self) -> bool:
        return self.mode == STRICT_BUDGET_MODE

    @property
    def scope(self) -> str:
        return "end_to_end_solver" if self.is_strict else "optimization_loop"

    @property
    def stopping_rule(self) -> str:
        if self.is_strict:
            return "deadline_aware_best_completed_incumbent"
        return "non_preemptive_work_unit_boundary"

    def start_search(self) -> float:
        if self.search_started_at is None:
            self.search_started_at = float(self.clock())
            if (
                self.mode == LEGACY_RESULTS_BUDGET_MODE
                and self.limit_seconds is not None
            ):
                self.deadline = self.search_started_at + self.limit_seconds
        return self.search_started_at

    def now(self) -> float:
        return float(self.clock())

    def expired(self, now: Optional[float] = None) -> bool:
        if self.deadline is None:
            return False
        current = self.now() if now is None else float(now)
        return current >= self.deadline

    def remaining(self, now: Optional[float] = None) -> float:
        if self.deadline is None:
            return math.inf
        current = self.now() if now is None else float(now)
        return max(0.0, self.deadline - current)

    def preprocessing_runtime(self, now: Optional[float] = None) -> float:
        end = (
            self.search_started_at
            if self.search_started_at is not None
            else (self.now() if now is None else float(now))
        )
        return max(0.0, end - self.run_started_at)

    def search_runtime(self, now: Optional[float] = None) -> float:
        if self.search_started_at is None:
            return 0.0
        end = self.now() if now is None else float(now)
        return max(0.0, end - self.search_started_at)

    def total_runtime(self, now: Optional[float] = None) -> float:
        end = self.now() if now is None else float(now)
        return max(0.0, end - self.run_started_at)

    def overshoot_seconds(self, now: Optional[float] = None) -> float:
        if self.limit_seconds is None:
            return 0.0
        return max(0.0, self.total_runtime(now) - self.limit_seconds)

    def metadata(self) -> dict:
        return {
            "budget_mode": self.mode,
            "budget_seconds": self.limit_seconds,
            "budget_scope": self.scope,
            "stopping_rule": self.stopping_rule,
            "preprocessing_included": self.is_strict,
            "postprocess_reserve_seconds": self.effective_reserve_seconds,
        }
