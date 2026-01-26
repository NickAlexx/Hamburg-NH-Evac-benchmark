# Path: backend\app\evacuation\core.py
import json
import os
import math
import random
import sys
import pyproj
from shapely.geometry import Point
from shapely.ops import transform
from openrouteservice import Client
from ..config import ORS_KEY

# Set random seed for reproducibility if needed.
random.seed()

# Global variables for simulation results and shared data
depots = []
facilities = []
durations_matrix = {}
demand_full = {}
pickup_nodes = []
deadlines = {}
n_depots = 1
n_facilities = 0
MAX_TRIPS_PER_BUS = 5
MAX_STOPS_PER_TRIP = 5

# Constants for both algorithms
PENALTY_FACTOR = 1e9
EXTRA_TRIP_PENALTY_FACTOR = 1e4
STOP_EMPTY_PENALTY = 1e9
STOP_FULL_PENALTY = 1e9
LATE_PENALTY = 1e6
DEPOT_OVERFILL_PENALTY = 1e7

# Initialize ORS client (only if a key is configured)
ors = None
if ORS_KEY:
    try:
        ors = Client(key=ORS_KEY)
    except Exception as e:
        print(f"Error initializing ORS client: {e}")

# Define static evacuation center (depot) parameters
buffer_meters = 1500  # Buffer distance in meters.
center_coords = (9.996754980861652, 53.49221335731889)  # (lon, lat)
center_label = "Evacuation Center"

def create_buffer_polygon(center_coords_tuple, buffer_m):
    """Utility function to create a buffer polygon."""
    global to_utm, to_wgs84
    try:
        center_point = Point(center_coords_tuple)
        utm_center = transform(to_utm, center_point)
        utm_buffer = utm_center.buffer(buffer_m)
        return transform(to_wgs84, utm_buffer)
    except Exception as e:
        print(f"Error creating buffer polygon: {e}")
        return None

# Prepare shapely transformations (WGS84 <-> UTM)
try:
    proj_wgs84 = pyproj.CRS("EPSG:4326")
    proj_utm = pyproj.CRS("EPSG:32632")  # UTM zone 32N (Hamburg region)
    to_utm = pyproj.Transformer.from_crs(proj_wgs84, proj_utm, always_xy=True).transform
    to_wgs84 = pyproj.Transformer.from_crs(proj_utm, proj_wgs84, always_xy=True).transform

    # Create a default buffer polygon for initialization if needed elsewhere
    buffer_polygon_wgs84 = create_buffer_polygon(center_coords, buffer_meters)
except Exception as e:
    print(f"Error setting up projection: {e}")
    buffer_polygon_wgs84 = None
    to_utm = None
    to_wgs84 = None

# Load facility data
def load_facility_data(buffer_polygon, buffer_dist):
    """Load and filter facilities within the buffer."""
    json_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "de_hh_up_vollstationaere_pflegeeinrichtungen_EPSG_4326.json")
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {json_file}. Total features: {len(data['features'])}")
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return []
        
    # Filter facilities within the buffer
    filtered_facilities = []
    if buffer_polygon:
        for feature in data["features"]:
            lon, lat = feature["geometry"]["coordinates"]
            p = Point(lon, lat)
            if p.within(buffer_polygon):
                filtered_facilities.append(feature)
        print(f"Number of facilities in {buffer_dist} m buffer: {len(filtered_facilities)}")
    else:
        # If buffer creation failed, just take the first few facilities
        filtered_facilities = data["features"][:10]
        print(f"Using first {len(filtered_facilities)} facilities (buffer creation failed)")
    
    return filtered_facilities

# Initialize problem data
def initialize_problem_data(evacuation_zones_input=None, buses_count=3, bus_capacity=80, default_evac_center_coords=None, buffer_meters_input=None):
    """Initialize problem data including depots, facilities, and duration matrix."""
    global depots, facilities, n_depots, n_facilities, durations_matrix, pickup_nodes, demand_full, deadlines, MAX_TRIPS_PER_BUS, MAX_STOPS_PER_TRIP
    
    # --- Use input values or fall back to defaults ---
    current_center_coords = default_evac_center_coords if default_evac_center_coords is not None else center_coords
    current_buffer_meters = buffer_meters_input if buffer_meters_input is not None else buffer_meters
    
    # --- Re-create buffer polygon based on current values ---
    local_buffer_polygon_wgs84 = create_buffer_polygon(current_center_coords, current_buffer_meters)
    
    # --- Setup Depots (Evacuation Zones) ---
    # The main center is now only for buffer calculation, not a depot itself.
    depots = []
    if evacuation_zones_input:
        for zone in evacuation_zones_input:
            depots.append({
                "label": zone["label"],
                "coords": tuple(zone["coords"]),
                "people": 0,
                "capacity": zone.get("capacity")  # Add capacity, defaults to None
            })

    if not depots:
        # If no user-defined depots are provided, the system cannot function.
        # The algorithms require at least one depot to return buses to.
        raise ValueError("Cannot run optimization without at least one user-defined evacuation zone (depot). Please add an evacuation center on the map.")
        
    n_depots = len(depots)

    # --- Setup Facilities (Care Facilities) ---
    filtered_facilities = load_facility_data(local_buffer_polygon_wgs84, current_buffer_meters)
    facilities = []
    node_coords = {}
    for feat in filtered_facilities:
        props = feat["properties"]
        street = props.get("adresse", "Unknown Street")
        city = props.get("ort", "Unknown City")
        people_count = props.get("plaetze", 1)
        facility_address = f"{street}, {city}"
        lon, lat = feat["geometry"]["coordinates"]
        facilities.append({
            "label": facility_address,
            "coords": (lon, lat),
            "people": people_count
        })
    n_facilities = len(facilities)

    # Initialize pickup nodes and demand
    pickup_nodes = list(range(n_facilities))
    demand_full = {i: facilities[i]["people"] for i in pickup_nodes}
    # Store coordinates in (lat, lon) format for first-leg travel calculations
    for idx in pickup_nodes:
        lon, lat = facilities[idx]["coords"]
        node_coords[idx] = (lat, lon)
    deadlines = {i: 120 for i in pickup_nodes}  # e.g., 120 min deadline

    # --- SIMPLIFIED & ROBUST INITIALIZATION LIMITS ---
    # MAX_TRIPS_PER_BUS is only used to generate the initial random population.
    # The algorithm can evolve solutions with more trips. A fixed, generous
    # upper bound is more robust than a complex heuristic.
    if n_facilities > 0:
        MAX_TRIPS_PER_BUS = 15 # A sensible, fixed upper bound for creating initial solutions.
        MAX_STOPS_PER_TRIP = int(math.sqrt(n_facilities) + 3)
    else:
        MAX_TRIPS_PER_BUS = 5
        MAX_STOPS_PER_TRIP = 5
    
    # Print values for debugging
    print(f"Core initialized with MAX_TRIPS_PER_BUS={MAX_TRIPS_PER_BUS}, MAX_STOPS_PER_TRIP={MAX_STOPS_PER_TRIP}")
    print(f"Core initialized with {n_facilities} facilities and {len(pickup_nodes)} pickup nodes")

    # --- Compute Combined Distance Matrix ---
    try:
        if not ors:
            raise Exception("ORS client not available")
            
        combined_coords = [d["coords"] for d in depots] + [f["coords"] for f in facilities]
        if not combined_coords:
            raise Exception("No coordinates available for distance matrix")
            
        matrix_result = ors.distance_matrix(
            locations=combined_coords,
            profile="driving-car",
            metrics=["distance", "duration"]
        )
        durations_matrix = {}
        duration_values = matrix_result["durations"]
        total_nodes = n_depots + n_facilities
        for i in range(total_nodes):
            for j in range(total_nodes):
                durations_matrix[(i, j)] = duration_values[i][j]
        print("Successfully retrieved combined distance/duration matrix from ORS.")
        print(f"Distance matrix size: {len(durations_matrix)} entries")
    except Exception as e:
        print(f"Error retrieving matrix from ORS: {e}")
        print("Using default travel times instead")
        
        # Create a simple distance matrix as fallback
        durations_matrix = {}
        total_nodes = n_depots + n_facilities
        for i in range(total_nodes):
            for j in range(total_nodes):
                # Use a simple calculation based on straight-line distance
                if i == j:
                    durations_matrix[(i, j)] = 0
                else:
                    # Get coordinates
                    if i < n_depots:
                        coord1 = depots[i]["coords"]
                    else:
                        coord1 = facilities[i - n_depots]["coords"]
                        
                    if j < n_depots:
                        coord2 = depots[j]["coords"]
                    else:
                        coord2 = facilities[j - n_depots]["coords"]
                        
                    # Calculate Euclidean distance (simplified)
                    dx = coord1[0] - coord2[0]
                    dy = coord1[1] - coord2[1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Convert to approximate seconds (1 degree ≈ 111 km, 40 km/h)
                    durations_matrix[(i, j)] = distance * 111000 * 60.0 / 40000
        
        print(f"Created fallback distance matrix with {len(durations_matrix)} entries")
        
    # Return all necessary data to ensure modules have synchronized data
    return {
        'depots': depots,
        'facilities': facilities, 
        'durations_matrix': durations_matrix,
        'max_trips_per_bus': MAX_TRIPS_PER_BUS,
        'max_stops_per_trip': MAX_STOPS_PER_TRIP,
        'pickup_nodes': pickup_nodes,
        'demand_full': demand_full,
        'deadlines': deadlines,
        'n_depots': n_depots,
        'n_facilities': n_facilities,
        'node_coords': node_coords,
    }

# Common utility function for both algorithms
def travel_time_for_trip(trip, bus_capacity):
    """
    Compute travel time (in minutes) for a trip.
    Each trip is a dict: {"start_depot": int, "stops": [facility indices], "end_depot": int}
    The trip time is: start_depot -> first stop -> ... -> last stop -> end_depot.
    Also adds 10 minutes for each stop to account for pickup/delivery time.
    """
    # Check if durations_matrix is None or empty to avoid the NoneType error
    if not durations_matrix:
        # Calculate a default travel time based on number of stops
        stops = trip.get("stops", [])  # Safely get stops with default empty list
        return 30.0 + len(stops) * 15.0  # 30 min base + 15 min per stop
    
    # Get values with safe defaults
    start_depot = trip.get("start_depot", 0)  # Default to depot 0 if missing
    end_depot = trip.get("end_depot", 0)      # Default to depot 0 if missing
    stops = trip.get("stops", [])             # Default to empty list if missing
    
    # Base travel time between locations
    try:
        if stops:
            time = durations_matrix.get((start_depot, n_depots + stops[0]), float('inf')) / 60.0
            for i in range(len(stops) - 1):
                time += durations_matrix.get((n_depots + stops[i], n_depots + stops[i+1]), float('inf')) / 60.0
            time += durations_matrix.get((n_depots + stops[-1], end_depot), float('inf')) / 60.0
            
            # Add 10 minutes per stop for pickup/delivery time
            time += len(stops) * 10.0
        else:
            time = durations_matrix.get((start_depot, end_depot), float('inf')) / 60.0
    except Exception as e:
        print(f"Error calculating travel time: {e}")
        # Fallback calculation
        time = 30.0 + len(stops) * 15.0
    
    # If time is infinite, provide a realistic fallback
    if time == float('inf') or time > 10000:
        time = 30.0 + len(stops) * 15.0  # Rough estimate
    
    return time

# Decode a solution to calculate its cost (used by both algorithms)
def decode_individual(individual, n_buses, bus_capacity):
    """
    Decode an individual.
    'individual' is a list of bus schedules (one per bus).
    Each bus schedule is a list of trips.
    Each trip is a dict: {"start_depot": int, "stops": [facility indices], "end_depot": int}
    For each bus, we enforce that the first trip starts at depot 0.
    """
    # Check if demand_full is None
    if demand_full is None:
        print("Warning: demand_full is None in decode_individual")
        return (PENALTY_FACTOR * n_facilities,)  # Return a high penalty
        
    # Create a copy of demand_full to track remaining demand
    try:
        remaining_demand = {i: demand_full.get(i, 0) for i in pickup_nodes}
    except Exception as e:
        print(f"Error initializing remaining_demand: {e}")
        remaining_demand = {}
        
    total_cost = 0.0
    finish_times = []
    bus_trip_start_times_list = []

    # Process only up to the requested number of buses
    for bus_idx, bus_schedule in enumerate(individual[:n_buses]):
        if not bus_schedule:
            finish_times.append(0.0)
            bus_trip_start_times_list.append([])
            continue

        # Enforce that the first trip starts at the static depot (index 0)
        bus_schedule[0]["start_depot"] = 0
        current_depot = 0
        bus_trip_start_times = []
        current_time = 0.0

        for trip in bus_schedule:
            # Ensure the trip's start depot is the current depot
            trip["start_depot"] = current_depot
            bus_trip_start_times.append(current_time)
            t_time = travel_time_for_trip(trip, bus_capacity)
            trip_return_time = current_time + t_time

            trip_load = 0
            extra_penalty_trip = 0.0
            remaining_capacity = bus_capacity
            stops = trip.get("stops", [])  # Safely get stops

            if stops:
                # Get the arrival time at the first stop
                arrival_time = current_time + durations_matrix.get((current_depot, n_depots + stops[0]), float('inf')) / 60.0
                
                if remaining_capacity > 0:
                    available = remaining_demand.get(stops[0], 0)
                    if available > 0:
                        pickup = min(available, remaining_capacity)
                        remaining_demand[stops[0]] -= pickup
                        trip_load += pickup
                        remaining_capacity -= pickup
                        deadline = deadlines.get(stops[0], float('inf'))
                        if arrival_time > deadline:
                            extra_penalty_trip += LATE_PENALTY * (arrival_time - deadline) * pickup
                    else:
                        total_cost += STOP_EMPTY_PENALTY
                else:
                    total_cost += STOP_FULL_PENALTY

                # Process subsequent stops
                for i in range(1, len(stops)):
                    arrival_time += durations_matrix.get((n_depots + stops[i-1], n_depots + stops[i]), float('inf')) / 60.0
                    if remaining_capacity <= 0:
                        total_cost += STOP_FULL_PENALTY
                        break
                        
                    available = remaining_demand.get(stops[i], 0)
                    if available > 0:
                        pickup = min(available, remaining_capacity)
                        remaining_demand[stops[i]] -= pickup
                        trip_load += pickup
                        remaining_capacity -= pickup
                        deadline = deadlines.get(stops[i], float('inf'))
                        if arrival_time > deadline:
                            extra_penalty_trip += LATE_PENALTY * (arrival_time - deadline) * pickup
                    else:
                        total_cost += STOP_EMPTY_PENALTY

            baseline_cost = trip_return_time * trip_load
            total_cost += baseline_cost + extra_penalty_trip

            current_time = trip_return_time
            # The bus finishes the trip at trip["end_depot"], which becomes the starting depot for the next trip.
            current_depot = trip["end_depot"]

        finish_times.append(current_time)
        bus_trip_start_times_list.append(bus_trip_start_times)

    # Extra penalty for idle finish times
    valid_bus_schedules = [s for s in individual[:n_buses] if s]
    if valid_bus_schedules:
        try:
            min_trip_count = min(len(s) for s in valid_bus_schedules)
            idle_finish_times = [
                finish_times[i]
                for i in range(len(finish_times))
                if individual[i] and len(individual[i]) == min_trip_count
            ]
            if idle_finish_times:
                T_idle = min(idle_finish_times)
                for i, bus_schedule in enumerate(individual[:n_buses]):
                    if bus_schedule and len(bus_schedule) > min_trip_count:
                        for trip_idx in range(min_trip_count, len(bus_schedule)):
                            if i < len(bus_trip_start_times_list) and trip_idx < len(bus_trip_start_times_list[i]):
                                start_time = bus_trip_start_times_list[i][trip_idx]
                                if start_time > T_idle:
                                    total_cost += EXTRA_TRIP_PENALTY_FACTOR * (start_time - T_idle)
        except Exception as e:
            print(f"Error calculating extra trip penalties: {e}")
            
    # Add penalty for unserved demand
    try:
        unserved = sum(remaining_demand.get(i, 0) for i in pickup_nodes)
        if unserved > 0:
            total_cost += PENALTY_FACTOR * unserved
    except Exception as e:
        print(f"Error calculating unserved demand: {e}")
        
    return (total_cost,)
