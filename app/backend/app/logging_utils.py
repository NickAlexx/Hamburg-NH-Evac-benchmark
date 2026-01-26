# Path: backend/app/logging_utils.py

import logging
from typing import Dict, Any, Optional, List

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_generation_metrics(generation: int, metrics: Dict[str, float]):
    """
    No-op placeholder kept for compatibility after removing run tracking.
    """
    return None


def start_run(algorithm_name: str, params: Dict[str, Any]) -> Optional[str]:
    """
    No-op placeholder kept for compatibility after removing run tracking.
    """
    return None


def end_run():
    """No-op placeholder kept for compatibility after removing run tracking."""
    return None


def log_evacuation_run(
    params: Dict[str, Any],
    facilities: list,
    overall_cost: float,
    solution_summary: Dict[str, Any],
    depots: list,
    ea_stats: Optional[Dict[str, Any]] = None,
    *args: Any,
    **kwargs: Any
) -> Optional[str]:
    """
    No-op placeholder kept for compatibility after removing run tracking.
    """
    return None
