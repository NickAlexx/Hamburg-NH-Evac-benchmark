import unittest

from app.backend.app.evacuation import visualization
from app.backend.app.evacuation.metrics import (
    _simulate_and_get_timings,
    compute_solution_metrics,
)
from app.backend.app.main import OptimizationPayload
from benchmark_scripts.run_paper_experiments import (
    _deserialize_problem_data,
    _serialize_problem_data,
)


class FacilityDeadlineCleanupTests(unittest.TestCase):
    def setUp(self):
        self.durations = {
            (0, 0): 0.0,
            (0, 1): 60.0,
            (1, 0): 60.0,
            (1, 1): 0.0,
        }
        self.depots = [{"label": "Shelter", "capacity": 10}]
        self.facilities = [{"label": "Facility", "people": 4}]
        self.individual = [[{
            "start_depot": 0,
            "stops": [(0, 4)],
            "end_depot": 0,
        }]]
        self.solution = [[{
            "start_depot": 0,
            "stops": [0],
            "end_depot": 0,
            "pickup_counts": {0: 4},
        }]]

    def test_central_simulation_has_no_lateness_outputs(self):
        simulation = _simulate_and_get_timings(
            individual=self.individual,
            n_depots=1,
            durations_matrix=self.durations,
            origin_by_bus=[{"kind": "depot", "index": 0}],
            cap_by_bus=[4],
            depots=self.depots,
            node_coords={0: (53.5, 10.0)},
            start_to_node_seconds=None,
            avg_speed_kmh=30.0,
            road_factor=1.25,
            use_dynamic_service_time=True,
        )

        self.assertNotIn("total_lateness_pm", simulation)
        self.assertNotIn("late_pickup_count", simulation)

    def test_metrics_and_timeline_do_not_report_pickup_deadlines(self):
        metrics = compute_solution_metrics(
            solution=self.solution,
            buses_count=1,
            n_depots=1,
            durations_matrix=self.durations,
            demand_full={0: 4},
            depots=self.depots,
            vehicles=[{
                "capacity": 4,
                "start": {"kind": "depot", "index": 0},
            }],
            use_dynamic_service_time=True,
        )
        timeline = visualization.simulate_solution_with_timeline(
            self.solution,
            1,
            [4],
            self.depots,
            self.facilities,
            1,
            self.durations,
            {0: 4},
            use_dynamic_service_time=True,
        )
        details = timeline[0][0]["details"]

        self.assertNotIn("lateness", metrics)
        self.assertNotIn("late_people", metrics["counts"])
        self.assertNotIn("late_fraction", metrics["counts"])
        self.assertFalse(any("on time" in detail or "late" in detail for detail in details))

    def test_public_api_and_cached_problem_data_drop_obsolete_fields(self):
        self.assertNotIn("lateness_penalty_factor", OptimizationPayload.model_fields)

        legacy_problem = {
            "depots": [],
            "facilities": [],
            "durations_matrix": {},
            "pickup_nodes": [0],
            "demand_full": {"0": 4},
            "deadlines": {"0": 120},
            "node_coords": {},
        }
        self.assertNotIn("deadlines", _deserialize_problem_data(legacy_problem))
        self.assertNotIn("deadlines", _serialize_problem_data(legacy_problem))


if __name__ == "__main__":
    unittest.main()
