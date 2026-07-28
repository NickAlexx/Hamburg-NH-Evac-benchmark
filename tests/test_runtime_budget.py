import unittest

from app.backend.app.evacuation.runtime_budget import (
    LEGACY_RESULTS_BUDGET_MODE,
    STRICT_BUDGET_MODE,
    RuntimeBudget,
)


class FakeClock:
    def __init__(self, value: float = 0.0):
        self.value = float(value)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


class RuntimeBudgetTests(unittest.TestCase):
    def test_default_mode_is_strict(self):
        budget = RuntimeBudget(limit_seconds=1.0, clock=FakeClock())
        self.assertEqual(budget.mode, STRICT_BUDGET_MODE)
        self.assertTrue(budget.is_strict)

    def test_strict_budget_starts_at_solver_entry_and_reserves_postprocessing(self):
        clock = FakeClock(100.0)
        budget = RuntimeBudget(
            limit_seconds=10.0,
            mode="strict",
            postprocess_reserve_seconds=2.0,
            clock=clock,
        )

        self.assertEqual(budget.deadline, 109.0)
        self.assertEqual(budget.effective_reserve_seconds, 1.0)

        clock.advance(3.0)
        budget.start_search()
        self.assertEqual(budget.preprocessing_runtime(), 3.0)
        self.assertEqual(budget.remaining(), 6.0)

        clock.advance(6.0)
        self.assertTrue(budget.expired())
        self.assertEqual(budget.total_runtime(), 9.0)
        self.assertEqual(budget.overshoot_seconds(), 0.0)

        metadata = budget.metadata()
        self.assertEqual(metadata["budget_scope"], "end_to_end_solver")
        self.assertTrue(metadata["preprocessing_included"])

    def test_legacy_results_protocol_starts_when_search_starts(self):
        clock = FakeClock(200.0)
        budget = RuntimeBudget(
            limit_seconds=5.0,
            mode=LEGACY_RESULTS_BUDGET_MODE,
            clock=clock,
        )

        clock.advance(7.0)
        self.assertFalse(budget.expired())
        self.assertIsNone(budget.deadline)

        budget.start_search()
        self.assertEqual(budget.deadline, 212.0)
        self.assertEqual(budget.preprocessing_runtime(), 7.0)

        clock.advance(5.0)
        self.assertTrue(budget.expired())
        self.assertEqual(budget.search_runtime(), 5.0)
        self.assertEqual(budget.metadata()["budget_scope"], "optimization_loop")

    def test_unknown_protocol_is_rejected(self):
        with self.assertRaises(ValueError):
            RuntimeBudget(limit_seconds=1.0, mode="unknown")

    def test_removed_paper_specific_name_is_rejected(self):
        with self.assertRaises(ValueError):
            RuntimeBudget(limit_seconds=1.0, mode="paper_2026")


if __name__ == "__main__":
    unittest.main()
