import contextlib
import io
import unittest

from pydantic import ValidationError

from app.backend.app.main import OptimizationPayload
from benchmark_scripts.run_paper_experiments import (
    _recorded_budget_mode,
    _require_matching_budget_mode,
    build_argument_parser,
)


class BudgetModeInterfaceTests(unittest.TestCase):
    def test_api_defaults_to_strict_and_accepts_legacy_results(self):
        self.assertEqual(OptimizationPayload().budget_mode, "strict")
        self.assertEqual(
            OptimizationPayload(
                budget_mode="legacy_results",
            ).budget_mode,
            "legacy_results",
        )

    def test_api_rejects_removed_paper_specific_name(self):
        with self.assertRaises(ValidationError):
            OptimizationPayload(budget_mode="paper_2026")

    def test_cli_defaults_to_strict_and_accepts_legacy_results(self):
        parser = build_argument_parser()
        self.assertEqual(parser.parse_args([]).budget_mode, "strict")
        self.assertEqual(
            parser.parse_args(
                ["--budget-mode", "legacy_results"],
            ).budget_mode,
            "legacy_results",
        )

    def test_cli_rejects_removed_paper_specific_name(self):
        parser = build_argument_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--budget-mode", "paper_2026"])

    def test_unlabelled_stored_result_is_classified_as_legacy(self):
        self.assertEqual(
            _recorded_budget_mode({"algorithm_stats": {}}),
            "legacy_results",
        )

    def test_resume_rejects_mixed_protocols(self):
        with self.assertRaisesRegex(RuntimeError, "instead of mixing protocols"):
            _require_matching_budget_mode(
                {"budget_mode": "legacy_results"},
                "strict",
                "run_1.json",
            )


if __name__ == "__main__":
    unittest.main()
