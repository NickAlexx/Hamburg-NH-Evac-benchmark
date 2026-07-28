# Path: backend\app\evacuation\visualization.py

# Try to import but if it fails, continue with fallbacks
try:
    from .core import n_depots, facilities, depots, durations_matrix, demand_full
except (ImportError, AttributeError):
    print("Warning: Failed to import one or more globals from core - using fallbacks")
    n_depots = 1
    facilities = []
    depots = []
    durations_matrix = {}
    demand_full = {}

def simulate_solution_with_timeline(solution, num_buses, bus_capacity, 
                                   depots_data=None, facilities_data=None, 
                                   n_depots_value=None, durations_data=None,
                                   demand_data=None,
                                   **kwargs):
    """
    Simulate the solution timeline for visualization.
    Only process up to num_buses buses.
    
    Now accepts explicit parameters to ensure data is available even if global variables are missing.
    """
    global depots, facilities, durations_matrix, n_depots, demand_full
    
    # Use explicitly passed parameters if available, fallback to globals
    local_depots = depots_data if depots_data is not None else depots
    local_facilities = facilities_data if facilities_data is not None else facilities
    local_n_depots = n_depots_value if n_depots_value is not None else n_depots
    local_durations = durations_data if durations_data is not None else durations_matrix
    local_demand = demand_data if demand_data is not None else demand_full
    
    # Extract service time params from kwargs
    service_params = {
        "use_dynamic_service_time": kwargs.get('use_dynamic_service_time', False),
        "service_time_base_min": kwargs.get('service_time_base_min', 3.0),
        "service_time_per_person_min": kwargs.get('service_time_per_person_min', 20.0 / 60.0),
    }

    # Initialize empty simulation data
    simulation_data = {}
    
    # Check if solution is None or empty
    if not solution:
        print("Warning: Empty solution provided to simulate_solution_with_timeline")
        return simulation_data
    
    # Create fallback depots if needed
    if not local_depots:
        print("Warning: depots is None or empty in simulate_solution_with_timeline - creating fallbacks")
        # Create fallback depots - at least one main depot
        local_depots = [{"label": "Main Depot", "coords": (0, 0), "people": 0}]
        # Add more depots based on the solution structure
        max_depot_id = 0
        for bus_schedule in solution:
            for trip in bus_schedule:
                max_depot_id = max(max_depot_id, trip.get("start_depot", 0), trip.get("end_depot", 0))
        
        # Create additional depots as needed
        for i in range(1, max_depot_id + 1):
            local_depots.append({"label": f"Depot {i}", "coords": (0, 0), "people": 0})
            
        local_n_depots = len(local_depots)
        print(f"Created {local_n_depots} fallback depots")
    
    # Create fallback facilities if needed
    if not local_facilities:
        print("Warning: facilities is None or empty - creating fallbacks")
        # Find the unique facility indices referenced in the solution
        used_facilities = set()
        for bus_schedule in solution:
            for trip in bus_schedule:
                used_facilities.update(trip.get("stops", []))
        
        # Create fallback facilities
        local_facilities = []
        for fac_idx in sorted(used_facilities):
            local_facilities.append({
                "label": f"Facility {fac_idx}",
                "coords": (0, 0), 
                "people": local_demand.get(fac_idx, 20)  # Default people count
            })
        print(f"Created {len(local_facilities)} fallback facilities")
    
    # Ensure n_depots is consistent with the length of depots
    local_n_depots = len(local_depots)
    
    # Safely create a copy of demand_full with fallbacks
    remaining_demand = {}
    if local_demand:
        for i in local_demand:
            remaining_demand[i] = local_demand.get(i, 0)
    else:
        # Create fallback demand data
        for i in range(len(local_facilities)):
            remaining_demand[i] = local_facilities[i].get("people", 20)
    
    # Process buses
    actual_buses = min(len(solution), num_buses)
    
    for bus_idx in range(actual_buses):
        bus_schedule = solution[bus_idx]
        simulation_data[bus_idx] = {}
        current_time = 0.0
        # Force starting depot for each bus to static depot (0)
        current_depot = 0

        # Determine capacity for the current bus
        current_bus_cap = 80 # default
        if isinstance(bus_capacity, list):
            if bus_idx < len(bus_capacity):
                current_bus_cap = bus_capacity[bus_idx]
        elif isinstance(bus_capacity, int):
            current_bus_cap = bus_capacity


        for trip_idx, trip in enumerate(bus_schedule):
            details = []
            details.append(f"Bus {bus_idx} Trip {trip_idx}")
            trip["start_depot"] = current_depot
            
            # Get travel time with fallback, passing service params
            t_time = travel_time_for_trip(trip, current_bus_cap, local_durations, local_n_depots, **service_params)
            bus_available_time = current_time + t_time

            # Calculate offload time to determine when evacuees actually arrive vs. when bus is free.
            # This logic must mirror travel_time_for_trip's offloading component.
            total_people_in_trip = sum(trip.get("pickup_counts", {}).values())
            offload_service_time = 0.0
            offload_needed = total_people_in_trip > 0 or (not trip.get("pickup_counts") and trip.get("stops"))

            if offload_needed:
                if service_params["use_dynamic_service_time"]:
                    offload_service_time = service_params["service_time_base_min"] + total_people_in_trip * service_params["service_time_per_person_min"]
                else:
                    # Static model: scales with bus capacity only
                    offload_service_time = 4.0 + 6.0 * (current_bus_cap / 40.0)
            
            evacuee_return_time = bus_available_time - offload_service_time
            
            # Safely access depot data
            start_depot_label = "Unknown Depot"
            if 0 <= trip['start_depot'] < len(local_depots):
                start_depot_label = local_depots[trip['start_depot']]['label']
                
            end_depot_label = "Unknown Depot"    
            if 0 <= trip['end_depot'] < len(local_depots):
                end_depot_label = local_depots[trip['end_depot']]['label']
            
            details.append(f"Departure from {start_depot_label}: {current_time:.2f} min")
            details.append(f"Arrival at {end_depot_label}: {evacuee_return_time:.2f} min")
            details.append(f"Bus busy for trip: {t_time:.2f} min")
            
            # Calculate stop time for clarity in details
            stop_time = 0.0
            if service_params["use_dynamic_service_time"] and "pickup_counts" in trip:
                total_people = sum(trip["pickup_counts"].values())
                stop_time = len(trip.get("stops", [])) * service_params["service_time_base_min"] + total_people * service_params["service_time_per_person_min"]
            else:
                num_stops = len(trip.get("stops", []))
                if num_stops > 0:
                    stop_time = num_stops * (4.0 + 6.0 * (current_bus_cap / 40.0))

            if stop_time > 0:
                details.append(f"Total time at stops: {stop_time:.2f} min")

            # Build route string using actual address names
            route_str = f"Depot: {start_depot_label}"
            for s in trip.get("stops", []):
                # Safely access facility data
                facility_label = f"Facility {s}"
                if s < len(local_facilities):
                    facility_label = local_facilities[s].get('label', f"Facility {s}")
                route_str += f" -> Facility: {facility_label}"
            route_str += f" -> Depot: {end_depot_label}"
            details.append("Route: " + route_str)

            # Check if we have pickup_counts in the trip data (from ea)
            has_pickup_counts = "pickup_counts" in trip
            
            rem_cap = current_bus_cap
            stops = trip.get("stops", [])
            if stops:
                # We need to track arrival time at each facility with the 10-minute stops included
                arrival_time = current_time
                
                # Calculate arrival time at first stop
                if local_durations:
                    key = (trip["start_depot"], local_n_depots + stops[0])
                    time_value = local_durations.get(key, 1800) / 60.0
                    arrival_time += time_value
                else:
                    arrival_time += 30.0  # Default travel time
                
                # Process first stop    
                if rem_cap > 0:
                    # If using ea with pickup_counts
                    if has_pickup_counts and stops[0] in trip["pickup_counts"]:
                        pickup = trip["pickup_counts"][stops[0]]
                        if pickup > 0:
                            rem_cap -= pickup
                            
                            # Get facility label
                            facility_label = f"Facility {stops[0]}"
                            if stops[0] < len(local_facilities):
                                facility_label = local_facilities[stops[0]].get('label', f"Facility {stops[0]}")
                                
                            details.append(f"Stop {facility_label}: picked up {pickup}")
                        else:
                            # Get facility label
                            facility_label = f"Facility {stops[0]}"
                            if stops[0] < len(local_facilities):
                                facility_label = local_facilities[stops[0]].get('label', f"Facility {stops[0]}")
                                
                            details.append(f"Stop {facility_label}: no evacuees")
                    # Traditional approach using remaining_demand
                    else:
                        available = remaining_demand.get(stops[0], 0)
                        if available > 0:
                            pickup = min(available, rem_cap)
                            remaining_demand[stops[0]] = remaining_demand.get(stops[0], 0) - pickup
                            rem_cap -= pickup
                            # Get facility label
                            facility_label = f"Facility {stops[0]}"
                            if stops[0] < len(local_facilities):
                                facility_label = local_facilities[stops[0]].get('label', f"Facility {stops[0]}")
                                
                            details.append(f"Stop {facility_label}: picked up {pickup}")
                        else:
                            # Get facility label
                            facility_label = f"Facility {stops[0]}"
                            if stops[0] < len(local_facilities):
                                facility_label = local_facilities[stops[0]].get('label', f"Facility {stops[0]}")
                                
                            details.append(f"Stop {facility_label}: no evacuees")
                
                # Add dynamic or static service time
                pickup_count_stop = trip.get("pickup_counts", {}).get(stops[0], 0) if has_pickup_counts else 0
                if service_params["use_dynamic_service_time"]:
                    arrival_time += service_params["service_time_base_min"] + pickup_count_stop * service_params["service_time_per_person_min"]
                else:
                    arrival_time += 4.0 + 6.0 * (current_bus_cap / 40.0)
                
                # Process remaining stops
                for i in range(1, len(stops)):
                    # Travel time to next stop
                    if local_durations:
                        key = (local_n_depots + stops[i-1], local_n_depots + stops[i])
                        time_value = local_durations.get(key, 1800) / 60.0
                        arrival_time += time_value
                    else:
                        arrival_time += 30.0  # Default travel time
                    
                    if rem_cap > 0:
                        # If using ea with pickup_counts
                        if has_pickup_counts and stops[i] in trip["pickup_counts"]:
                            pickup = trip["pickup_counts"][stops[i]]
                            if pickup > 0:
                                rem_cap -= pickup
                                
                                # Get facility label
                                facility_label = f"Facility {stops[i]}"
                                if stops[i] < len(local_facilities):
                                    facility_label = local_facilities[stops[i]].get('label', f"Facility {stops[i]}")
                                    
                                details.append(f"Stop {facility_label}: picked up {pickup}")
                            else:
                                # Get facility label
                                facility_label = f"Facility {stops[i]}"
                                if stops[i] < len(local_facilities):
                                    facility_label = local_facilities[stops[i]].get('label', f"Facility {stops[i]}")
                                    
                                details.append(f"Stop {facility_label}: no evacuees")
                        # Traditional approach using remaining_demand
                        else:
                            available = remaining_demand.get(stops[i], 0)
                            if available > 0:
                                pickup = min(available, rem_cap)
                                remaining_demand[stops[i]] = remaining_demand.get(stops[i], 0) - pickup
                                rem_cap -= pickup
                                # Get facility label
                                facility_label = f"Facility {stops[i]}"
                                if stops[i] < len(local_facilities):
                                    facility_label = local_facilities[stops[i]].get('label', f"Facility {stops[i]}")
                                    
                                details.append(f"Stop {facility_label}: picked up {pickup}")
                            else:
                                # Get facility label
                                facility_label = f"Facility {stops[i]}"
                                if stops[i] < len(local_facilities):
                                    facility_label = local_facilities[stops[i]].get('label', f"Facility {stops[i]}")
                                    
                                details.append(f"Stop {facility_label}: no evacuees")
                    
                    # Add dynamic or static service time
                    pickup_count_stop = trip.get("pickup_counts", {}).get(stops[i], 0) if has_pickup_counts else 0
                    if service_params["use_dynamic_service_time"]:
                        arrival_time += service_params["service_time_base_min"] + pickup_count_stop * service_params["service_time_per_person_min"]
                    else:
                        arrival_time += 4.0 + 6.0 * (current_bus_cap / 40.0)
            else:
                details.append("No stops on this trip.")

            if rem_cap == current_bus_cap:
                details.append("No pickups on this trip.")

            simulation_data[bus_idx][trip_idx] = {
                'details': details,
                'departure': current_time,
                'return': evacuee_return_time, # Evacuee arrival time
                'trip_time': t_time
            }
            current_time = bus_available_time # Bus is available after offloading
            current_depot = trip["end_depot"]

    return simulation_data

def generate_facility_timeline_features(simulation_solution):
    # For simplicity, return an empty list. Implement as needed.
    return []

def create_solution_summary(solution, buses_count, simulation_data):
    """Helper function to create a summary of the solution for logging."""
    summary = {"buses": []}
    
    # Check if solution is None
    if not solution:
        print("Warning: Empty solution provided to create_solution_summary")
        return summary
    
    for bus_idx, bus_schedule in enumerate(solution):
        if bus_idx >= buses_count:
            break  # Only include buses that were actually used
            
        bus_data = {"trips": []}
        for trip_idx, trip in enumerate(bus_schedule):
            trip_data = {
                "start_depot": trip["start_depot"],
                "stops": trip.get("stops", []),  # Safe get with default
                "end_depot": trip["end_depot"]
            }
            
            # Add pickup_counts if available (for EA2)
            if "pickup_counts" in trip:
                trip_data["pickup_counts"] = trip["pickup_counts"]
            
            # Add simulation data if available
            trip_sim = None
            # Handle both string and integer keys
            if str(bus_idx) in simulation_data and str(trip_idx) in simulation_data[str(bus_idx)]:
                trip_sim = simulation_data[str(bus_idx)][str(trip_idx)]
            elif bus_idx in simulation_data and trip_idx in simulation_data[bus_idx]:
                trip_sim = simulation_data[bus_idx][trip_idx]
                
            if trip_sim:
                trip_data["departure"] = trip_sim["departure"]
                trip_data["return"] = trip_sim["return"]
                trip_data["trip_time"] = trip_sim["trip_time"]
                trip_data["details"] = trip_sim["details"]
            
            bus_data["trips"].append(trip_data)
        
        summary["buses"].append(bus_data)
    
    return summary

# Helper function for trip time calculation with explicit parameter options
def travel_time_for_trip(trip, bus_capacity, durations_data=None, n_depots_value=None, **kwargs):
    """
    Compute travel time (in minutes) for a trip. This now represents the total time
    a bus is occupied, including travel, boarding, and offloading.
    
    Each trip is a dict: {"start_depot": int, "stops": [facility indices], "end_depot": int}
    
    Now accepts explicit parameters to ensure data is available even if global variables are missing.
    """
    global durations_matrix, n_depots
    
    # Use passed parameters if available, fallback to globals
    local_durations = durations_data if durations_data is not None else durations_matrix
    local_n_depots = n_depots_value if n_depots_value is not None else n_depots
    
    # Extract service time params
    use_dynamic_service_time = kwargs.get('use_dynamic_service_time', False)
    service_time_base_min = kwargs.get('service_time_base_min', 3.0)
    service_time_per_person_min = kwargs.get('service_time_per_person_min', 20.0 / 60.0)

    # Check if durations_matrix is None or empty to avoid the NoneType error
    if not local_durations:
        # Calculate a default travel time based on number of stops
        stops = trip.get("stops", [])  # Safely get stops with default empty list
        return 30.0 + len(stops) * 15.0  # 30 min base + 15 min per stop
    
    # Get a safe value for n_depots
    local_n_depots = local_n_depots if local_n_depots is not None else 1
    
    start_depot = trip.get("start_depot", 0)  # Default to depot 0 if missing
    end_depot = trip.get("end_depot", 0)      # Default to depot 0 if missing
    stops = trip.get("stops", [])             # Default to empty list if missing
    
    time = 0.0
    # Base travel time between locations
    if stops:
        try:
            time += local_durations.get((start_depot, local_n_depots + stops[0]), float('inf')) / 60.0
            for i in range(len(stops) - 1):
                time += local_durations.get((local_n_depots + stops[i], local_n_depots + stops[i+1]), float('inf')) / 60.0
            time += local_durations.get((local_n_depots + stops[-1], end_depot), float('inf')) / 60.0
            
        except (IndexError, TypeError) as e:
            print(f"Error calculating travel time: {e}")
            # Fallback time calculation
            time = 30.0 + len(stops) * 15.0
    else:
        time = local_durations.get((start_depot, end_depot), float('inf')) / 60.0

    # Add boarding service time at pickup stops
    total_people_in_trip = sum(trip.get("pickup_counts", {}).values())
    if use_dynamic_service_time and "pickup_counts" in trip:
        service_time_stops = len(stops) * service_time_base_min
        service_time_people = total_people_in_trip * service_time_per_person_min
        time += service_time_stops + service_time_people
    else:
        # Static model: boarding time scales with bus capacity.
        num_stops = len(stops)
        if num_stops > 0:
            time += num_stops * (4.0 + 6.0 * (bus_capacity / 40.0))
    
    # Add offloading service time at the destination depot
    offload_needed = total_people_in_trip > 0 or (not trip.get("pickup_counts") and stops)
    if offload_needed:
        if use_dynamic_service_time and "pickup_counts" in trip:
            offload_service_time = service_time_base_min + total_people_in_trip * service_time_per_person_min
            time += offload_service_time
        else:
            # Static model: offloading time scales with bus capacity.
            offload_service_time = 4.0 + 6.0 * (bus_capacity / 40.0)
            time += offload_service_time

    # If time is infinite, provide a realistic fallback
    if time == float('inf') or time > 10000:
        time = 30.0 + len(stops) * 15.0  # Rough estimate
    
    return time
