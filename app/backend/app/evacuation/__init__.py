# Path: backend/app/evacuation/__init__.py

"""
Lightweight package initializer for app.evacuation.

Avoid importing heavy/optional dependencies at import time so tests that only
need specific modules (e.g., baselines) don't require optional packages like deap.
"""

# Export the main algorithm function (best-effort; optional if deps missing)
try:
    from .ea import run_evolutionary_algorithm  # noqa: F401
except Exception:
    run_evolutionary_algorithm = None  # type: ignore

# Additional exports that might be needed elsewhere
try:
    from .visualization import simulate_solution_with_timeline, generate_facility_timeline_features  # noqa: F401
except Exception:
    simulate_solution_with_timeline = None  # type: ignore
    generate_facility_timeline_features = None  # type: ignore
