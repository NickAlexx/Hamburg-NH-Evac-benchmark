# Path: backend\app\evacuation\algorithm_interface.py
# Path: backend/app/evacuation/algorithm_interface.py

"""
Interface definition for evacuation routing algorithms.

This module defines the common interface that all evacuation routing algorithm 
implementations should follow to ensure consistency and interchangeability.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union

# Import common utilities
try:
    from .core import (
        initialize_problem_data, travel_time_for_trip, decode_individual,
        PENALTY_FACTOR, EXTRA_TRIP_PENALTY_FACTOR, STOP_EMPTY_PENALTY, STOP_FULL_PENALTY, LATE_PENALTY
    )
except Exception:
    # Degrade gracefully when optional geo deps are missing (tests can patch initialize_problem)
    initialize_problem_data = None  # type: ignore
    travel_time_for_trip = None  # type: ignore
    decode_individual = None  # type: ignore
    PENALTY_FACTOR = 1e9  # type: ignore
    EXTRA_TRIP_PENALTY_FACTOR = 1e4  # type: ignore
    STOP_EMPTY_PENALTY = 1e9  # type: ignore
    STOP_FULL_PENALTY = 1e9  # type: ignore
    LATE_PENALTY = 1e6  # type: ignore
from . import visualization
from ..logging_utils import log_evacuation_run


# Type definitions
# Note: Using Dict instead of TypedDict for better compatibility
Trip = Dict[str, Any]  # {"start_depot": int, "stops": List[int], "end_depot": int}
Location = Dict[str, Any]  # {"label": str, "coords": Tuple[float, float], "people": int}
TripSimulation = Dict[str, Any]  # {"details": List[str], "departure": float, "return": float, "trip_time": float}
SimulationData = Dict[str, Dict[str, TripSimulation]]
AlgorithmResult = Dict[str, Any]


class EvacuationAlgorithm(ABC):
    """
    Abstract base class for evacuation routing algorithms.
    
    This class defines the interface that all evacuation routing algorithm
    implementations should follow.
    """

    @abstractmethod
    def run(self, 
            evacuation_zones_input: Optional[List[Dict[str, Any]]] = None,
            buses_count: int = 3,
            bus_capacity: int = 80,
            **algorithm_specific_params) -> AlgorithmResult:
        """
        Run the evacuation optimization algorithm.
        
        Parameters:
        - evacuation_zones_input: List of user-defined evacuation zones (optional)
        - buses_count: Number of buses to use
        - bus_capacity: Capacity of each bus
        - **algorithm_specific_params: Algorithm-specific parameters
        
        Returns:
        - Dictionary with results including best solution, cost, and simulation data
        """
        pass

    @staticmethod
    def initialize_problem(evacuation_zones_input, buses_count, bus_capacity, default_evac_center_coords=None, buffer_meters=None):
        """
        Initialize the problem data using the shared core module.
        
        This method ensures all algorithms initialize data consistently.
        
        Returns:
        - Problem data dictionary with all necessary variables
        """
        # Initialize problem data
        if initialize_problem_data is None:
            # In lightweight test environments, this method is commonly patched.
            raise RuntimeError("initialize_problem_data unavailable; patch EvacuationAlgorithm.initialize_problem in tests.")
        problem_data = initialize_problem_data(
            evacuation_zones_input, buses_count, bus_capacity,
            default_evac_center_coords=default_evac_center_coords, buffer_meters_input=buffer_meters
        )
        
        # Update global variables in relevant modules
        from . import core
        core.depots = problem_data['depots']
        core.facilities = problem_data['facilities']
        core.durations_matrix = problem_data['durations_matrix']
        core.pickup_nodes = problem_data['pickup_nodes']
        core.MAX_TRIPS_PER_BUS = problem_data['max_trips_per_bus']
        core.MAX_STOPS_PER_TRIP = problem_data['max_stops_per_trip']
        core.n_depots = problem_data['n_depots']
        core.n_facilities = problem_data['n_facilities']
        core.demand_full = problem_data['demand_full']
        core.deadlines = problem_data['deadlines']
        
        # Also update visualization module
        visualization.depots = problem_data['depots']
        visualization.facilities = problem_data['facilities']
        visualization.durations_matrix = problem_data['durations_matrix']
        visualization.n_depots = problem_data['n_depots']
        visualization.demand_full = problem_data['demand_full']
        visualization.deadlines = problem_data['deadlines']
        
        return problem_data

    @staticmethod
    def create_simulation_data(best_solution, buses_count, bus_capacity, depots, facilities, n_depots, 
                              durations_matrix, demand_full, deadlines, **kwargs):
        """
        Create simulation data from the best solution for visualization.
        
        This ensures all algorithms produce consistent simulation data.
        """
        return visualization.simulate_solution_with_timeline(
            best_solution, buses_count, bus_capacity, depots, facilities, 
            n_depots, durations_matrix, demand_full, deadlines, **kwargs
        )

    @staticmethod
    def create_result_object(algorithm_name: str, overall_cost: float, best_solution: List[List[Trip]],
                           simulation_data: SimulationData, depots: List[Location], 
                           facilities: List[Location], buses_count: int, bus_capacity: int) -> AlgorithmResult:
        """
        Create a standardized result object.
        
        This ensures all algorithms return results in a consistent format.
        """
        return {
            "overall_cost": overall_cost,
            "best_solution": best_solution,
            "simulation_data": simulation_data,
            "facility_timeline_features": [],
            "depots": depots,
            "facilities": facilities,
            "num_buses": buses_count,
            "bus_capacity": bus_capacity,
            "algorithm": algorithm_name,
            "timestamp": None,  # Will be added by the API endpoint
            "logs_directory": None  # Will be added by the API endpoint
        }

    @staticmethod
    def log_algorithm_run(algorithm_name: str, params: Dict[str, Any], facilities: List[Location],
                         overall_cost: float, solution_summary: Dict[str, Any], 
                         depots: List[Location], algorithm_stats: Optional[Dict[str, Any]] = None) -> str:
        """
        Log the algorithm run details.
        
        This ensures all algorithms log their runs in a consistent format.
        """
        run_params = {
            "algorithm": algorithm_name,
            **params  # Include all algorithm-specific parameters
        }
        
        return log_evacuation_run(
            params=run_params,
            facilities=facilities,
            overall_cost=overall_cost,
            solution_summary=solution_summary,
            depots=depots,
            ea_stats=algorithm_stats  # This param name is used for all algorithm stats
        )


# Template for functional interface (for reference only - implementations should adapt this)
def algorithm_function_template(
    evacuation_zones_input: Optional[List[Dict[str, Any]]] = None,
    buses_count: int = 3,
    bus_capacity: int = 80,
    **algorithm_specific_params
) -> AlgorithmResult:
    """
    Template showing the expected structure for function-based algorithm implementations.
    
    This is a reference template. Actual implementations need to fill in the core algorithm logic.
    
    Parameters:
    - evacuation_zones_input: List of user-defined evacuation zones (optional)
    - buses_count: Number of buses to use
    - bus_capacity: Capacity of each bus
    - **algorithm_specific_params: Algorithm-specific parameters
    
    Returns:
    - Dictionary with results including best solution, cost, and simulation data
    """
    # 1. Initialize problem data
    problem_data = EvacuationAlgorithm.initialize_problem(
        evacuation_zones_input, buses_count, bus_capacity
    )
    
    # Access needed variables from problem_data
    depots = problem_data['depots']
    facilities = problem_data['facilities']
    durations_matrix = problem_data['durations_matrix'] 
    n_depots = problem_data['n_depots']
    n_facilities = problem_data['n_facilities']
    demand_full = problem_data['demand_full']
    deadlines = problem_data['deadlines']
    
    # 2. Algorithm implementation would go here
    # best_solution = ...
    # overall_cost = ...
    
    # THIS IS JUST PLACEHOLDER CODE - ACTUAL IMPLEMENTATION NEEDED
    best_solution = []  # Placeholder
    overall_cost = 0.0  # Placeholder
    
    # 3. Create simulation data
    simulation_data = EvacuationAlgorithm.create_simulation_data(
        best_solution, buses_count, bus_capacity, depots, facilities,
        n_depots, durations_matrix, demand_full, deadlines
    )
    
    # 4. Create solution summary for logging
    solution_summary = visualization.create_solution_summary(
        best_solution, buses_count, simulation_data
    )
    
    # 5. Create standardized result
    result = EvacuationAlgorithm.create_result_object(
        "example_algorithm", overall_cost, best_solution, simulation_data,
        depots, facilities, buses_count, bus_capacity
    )
    
    # 6. Log the run
    algorithm_params = {
        "buses_count": buses_count, 
        "bus_capacity": bus_capacity,
        **algorithm_specific_params
    }
    
    EvacuationAlgorithm.log_algorithm_run(
        "example_algorithm", 
        algorithm_params,
        facilities, 
        overall_cost, 
        solution_summary, 
        depots, 
        algorithm_stats=None  # Replace with actual algorithm stats
    )
    
    return result