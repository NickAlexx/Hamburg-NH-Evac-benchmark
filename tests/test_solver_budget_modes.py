import contextlib
import copy
import io
import json
import random
import time
import unittest

import numpy as np

from app.backend.app.evacuation.alns_algorithm import run_alns_algorithm
from app.backend.app.evacuation.baselines.pendelverkehr import (
    PendelverkehrShuttleAlgorithm,
)
from app.backend.app.evacuation.ea import run_evolutionary_algorithm


def tiny_problem():
    return {
        "depots": [
            {
                "label": "D0",
                "coords": (10.0, 53.5),
                "people": 0,
                "capacity": 20,
            }
        ],
        "facilities": [
            {"label": "F0", "coords": (10.01, 53.51), "people": 4},
            {"label": "F1", "coords": (10.02, 53.52), "people": 4},
        ],
        "durations_matrix": {
            (i, j): (0.0 if i == j else 60.0 + 10.0 * i + 5.0 * j)
            for i in range(3)
            for j in range(3)
        },
        "max_trips_per_bus": 4,
        "max_stops_per_trip": 2,
        "pickup_nodes": [0, 1],
        "demand_full": {0: 4, 1: 4},
        "n_depots": 1,
        "n_facilities": 2,
        "node_coords": {0: (53.51, 10.01), 1: (53.52, 10.02)},
    }


class SolverBudgetModeTests(unittest.TestCase):
    def setUp(self):
        self.common = {
            "buses_count": 2,
            "bus_capacity": 4,
            "vehicles": [
                {"capacity": 4, "start_depot": 0},
                {"capacity": 4, "start_depot": 0},
            ],
            "use_dynamic_service_time": True,
        }

    def _run_ea(
        self,
        *,
        mode,
        time_limit,
        generations,
        use_local_search=False,
    ):
        random.seed(7)
        np.random.seed(7)
        local_search_params = {}
        if use_local_search:
            local_search_params = {
                "ls_burn_in": 0,
                "ls_every": 1,
                "ls_offspring_prob": 0.0,
                "ls_max_elites": 1,
                "ls_params": {
                    "max_iterations": 1,
                    "time_limit_seconds": 0.02,
                    "micro_batch_calls": 1,
                },
            }
        with contextlib.redirect_stdout(io.StringIO()):
            return run_evolutionary_algorithm(
                **self.common,
                precomputed_problem_data=copy.deepcopy(tiny_problem()),
                population_size=8,
                generations=generations,
                crossover_rate=0.8,
                mutation_rate=0.2,
                tournament_size=2,
                use_local_search=use_local_search,
                early_stopping_generations=max(100, generations),
                time_limit_seconds=time_limit,
                budget_mode=mode,
                postprocess_reserve_seconds=0.1,
                **local_search_params,
            )

    def _run_alns(self, *, mode, time_limit, max_iterations):
        random.seed(7)
        np.random.seed(7)
        with contextlib.redirect_stdout(io.StringIO()):
            return run_alns_algorithm(
                **self.common,
                precomputed_problem_data=copy.deepcopy(tiny_problem()),
                seed=7,
                time_limit_seconds=time_limit,
                budget_mode=mode,
                postprocess_reserve_seconds=0.1,
                alns_config={
                    "max_iterations": max_iterations,
                    "use_memetic_polish": False,
                    "initial_shake_cycles": 0,
                    "sa_samples_for_T0": 0,
                    "log_every_iterations": 0,
                    "log_every_seconds": 0,
                    "stall_seconds": 9999,
                },
            )

    def assert_feasible(self, result):
        totals = {0: 0, 1: 0}
        for schedule in result["best_solution"]:
            previous_end = None
            for trip in schedule:
                if previous_end is not None:
                    self.assertEqual(trip["start_depot"], previous_end)
                load = sum(int(v) for v in trip["pickup_counts"].values())
                self.assertLessEqual(load, 4)
                for node, quantity in trip["pickup_counts"].items():
                    totals[int(node)] += int(quantity)
                previous_end = trip["end_depot"]
        self.assertEqual(totals, {0: 4, 1: 4})
        json.dumps(result)

    def test_modes_preserve_ea_behavior_when_the_deadline_does_not_bind(self):
        strict = self._run_ea(mode="strict", time_limit=5.0, generations=2)
        legacy = self._run_ea(
            mode="legacy_results",
            time_limit=5.0,
            generations=2,
        )

        self.assertEqual(strict["best_solution"], legacy["best_solution"])
        self.assertEqual(strict["overall_cost"], legacy["overall_cost"])
        self.assertEqual(strict["budget_mode"], "strict")
        self.assertEqual(legacy["budget_mode"], "legacy_results")
        self.assert_feasible(strict)

    def test_modes_preserve_alns_behavior_when_the_deadline_does_not_bind(self):
        strict = self._run_alns(
            mode="strict",
            time_limit=5.0,
            max_iterations=5,
        )
        legacy = self._run_alns(
            mode="legacy_results",
            time_limit=5.0,
            max_iterations=5,
        )

        self.assertEqual(strict["best_solution"], legacy["best_solution"])
        self.assertEqual(strict["best_fitness"], legacy["best_fitness"])
        self.assertEqual(strict["budget_mode"], "strict")
        self.assertEqual(legacy["budget_mode"], "legacy_results")
        self.assert_feasible(strict)

    def test_modes_preserve_memetic_behavior_when_the_deadline_does_not_bind(self):
        strict = self._run_ea(
            mode="strict",
            time_limit=5.0,
            generations=2,
            use_local_search=True,
        )
        legacy = self._run_ea(
            mode="legacy_results",
            time_limit=5.0,
            generations=2,
            use_local_search=True,
        )

        self.assertEqual(strict["best_solution"], legacy["best_solution"])
        self.assertEqual(strict["overall_cost"], legacy["overall_cost"])
        self.assert_feasible(strict)

    def test_dispatcher_reports_strict_budget_metadata(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = PendelverkehrShuttleAlgorithm().run(
                **self.common,
                precomputed_problem_data=copy.deepcopy(tiny_problem()),
                time_limit_seconds=1.0,
                budget_mode="strict",
                postprocess_reserve_seconds=0.1,
            )

        stats = result["algorithm_stats"]
        self.assertEqual(result["budget_mode"], "strict")
        self.assertEqual(stats["budget_scope"], "end_to_end_solver")
        self.assertTrue(stats["budget_adhered"])
        self.assertLessEqual(stats["total_runtime"], 1.0)
        self.assert_feasible(result)

    def test_strict_ea_stops_within_its_end_to_end_budget(self):
        wall_started = time.perf_counter()
        result = self._run_ea(
            mode="strict",
            time_limit=0.25,
            generations=100_000,
        )
        wall_runtime = time.perf_counter() - wall_started
        stats = result["algorithm_stats"]

        self.assertTrue(stats["stopped_by_time_limit"])
        self.assertTrue(stats["budget_adhered"])
        self.assertLessEqual(stats["total_runtime"], 0.25)
        self.assertLess(wall_runtime, 1.0)
        self.assert_feasible(result)

    def test_strict_alns_stops_within_its_end_to_end_budget(self):
        wall_started = time.perf_counter()
        result = self._run_alns(
            mode="strict",
            time_limit=0.25,
            max_iterations=1_000_000,
        )
        wall_runtime = time.perf_counter() - wall_started
        stats = result["algorithm_stats"]

        self.assertTrue(stats["stopped_by_time_limit"])
        self.assertTrue(stats["budget_adhered"])
        self.assertLessEqual(stats["total_runtime"], 0.25)
        self.assertLess(wall_runtime, 1.0)
        self.assert_feasible(result)


if __name__ == "__main__":
    unittest.main()
