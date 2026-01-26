# Path: backend\app\evacuation\utils.py

import copy
import math
import random

# Make a more robust import
try:
    from .core import travel_time_for_trip, n_depots
except (ImportError, AttributeError):
    # Define a fallback travel_time_for_trip function if import fails
    print("Warning: Failed to import travel_time_for_trip from core, using fallback")
    def travel_time_for_trip(trip, bus_capacity):
        stops = trip.get("stops", [])
        return 30.0 + len(stops) * 15.0  # 30 min base + 15 min per stop
    n_depots = 1

def local_search_improvement(individual, n_buses, bus_capacity):
    """
    Apply local search to improve an individual solution by finding better stop orderings.
    This can significantly reduce travel times and improve overall solution quality.
    """
    # Check if individual is None or empty
    if not individual:
        print("Warning: Empty solution provided to local_search_improvement")
        return []
    
    try:
        improved = copy.deepcopy(individual)
        
        # Examine each bus schedule
        for bus_idx in range(min(len(improved), n_buses)):
            bus_schedule = improved[bus_idx]
            
            # For each trip in the schedule
            for trip_idx in range(len(bus_schedule)):
                trip = bus_schedule[trip_idx]
                
                # Only try to optimize if the trip has multiple stops
                if len(trip.get("stops", [])) > 2:  # Safely get stops
                    try:
                        best_order = trip["stops"].copy()
                        trip_copy = copy.deepcopy(trip)
                        best_time = travel_time_for_trip(trip_copy, bus_capacity)
                        
                        # Try various permutations - limit to prevent excessive computation
                        max_iterations = min(20, math.factorial(len(trip["stops"])))
                        for _ in range(max_iterations):
                            # Create a new permutation by swapping two random positions
                            new_order = trip["stops"].copy()
                            i, j = random.sample(range(len(new_order)), 2)
                            new_order[i], new_order[j] = new_order[j], new_order[i]
                            
                            # Test if this new ordering reduces travel time
                            trip_copy["stops"] = new_order
                            new_time = travel_time_for_trip(trip_copy, bus_capacity)
                            
                            if new_time < best_time:
                                best_time = new_time
                                best_order = new_order.copy()
                        
                        # Apply the best ordering found
                        trip["stops"] = best_order
                    except Exception as e:
                        print(f"Error optimizing trip stops: {e}")
                    
                # Try to optimize end depot selection for each trip
                if len(trip.get("stops", [])) > 0:
                    try:
                        current_end = trip["end_depot"]
                        best_end = current_end
                        best_time = travel_time_for_trip(trip, bus_capacity)
                        
                        # Try each possible end depot
                        num_depots = n_depots if n_depots is not None else 1
                        
                        for depot_idx in range(num_depots):
                            if depot_idx != current_end:
                                trip["end_depot"] = depot_idx
                                new_time = travel_time_for_trip(trip, bus_capacity)
                                
                                if new_time < best_time:
                                    best_time = new_time
                                    best_end = depot_idx
                        
                        # Apply the best end depot found
                        trip["end_depot"] = best_end
                        
                        # If this isn't the last trip, we need to update the next trip's start depot
                        if trip_idx < len(bus_schedule) - 1:
                            bus_schedule[trip_idx + 1]["start_depot"] = best_end
                    except Exception as e:
                        print(f"Error optimizing end depot: {e}")
        
        return improved
    except Exception as e:
        print(f"Error in local_search_improvement: {e}")
        # Return the original solution if something goes wrong
        return individual