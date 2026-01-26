# Path: backend/app/main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
from dotenv import load_dotenv
import os
from datetime import datetime
from threading import Lock, Thread
from uuid import uuid4
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Tuple, Any, Literal

# Import the EA2 algorithm runner AND the baseline
from .evacuation.ea import run_evolutionary_algorithm
from .evacuation.baselines.pendelverkehr import PendelverkehrShuttleAlgorithm
# --- NEW: Import ALNS ---
from .evacuation.alns_algorithm import ALNSEvacuationAlgorithm
# NEW IMPORTS for pre-processing custom coordinates
from .evacuation.core import create_buffer_polygon, load_facility_data, center_coords as default_center_coords, buffer_meters as default_buffer_meters
from openrouteservice import Client as ORSClient

# Load environment variables from a .env file in the current directory.
load_dotenv()
ORS_KEY = os.getenv("ORS_KEY")

app = FastAPI()

# --- In-memory optimization job store (for long-running runs) ---
_optimization_jobs_lock = Lock()
_optimization_run_lock = Lock()
_optimization_jobs: dict[str, dict[str, Any]] = {}

# --- Pydantic Models for Request Body ---

class UserEvacCenter(BaseModel):
    """Defines a user-specified evacuation center."""
    label: str = Field(..., example="User-defined Center 1")
    coords: Tuple[float, float] = Field(..., example=[10.05, 53.56])
    capacity: Optional[int] = Field(None, description="Maximum number of people this center can hold.", gt=0)

# NEW: Models for heterogeneous fleet
class StartCoordinate(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)

class Vehicle(BaseModel):
    id: Optional[str] = None
    capacity: int = Field(..., gt=0)
    start_depot: Optional[int] = Field(None, ge=0)
    start_node: Optional[int] = Field(None, ge=0)
    start_coord: Optional[StartCoordinate] = None

    @validator('start_node', 'start_coord', always=True)
    def check_one_start_type(cls, v, values):
        starts_defined = sum(
            x is not None for x in 
            [values.get('start_depot'), v, values.get('start_coord')]
        )
        if starts_defined > 1:
            raise ValueError("Only one of start_depot, start_node, or start_coord may be specified.")
        return v

class OptimizationPayload(BaseModel):
    """Defines the request body for running an optimization."""
    algorithm_choice: Literal["ea2", "ea2_memetic", "pendelverkehr", "alns"] = Field(
        "ea2_memetic",
        description="The algorithm to run: 'ea2', 'ea2_memetic', 'pendelverkehr', or 'alns'."
    )
    evacuation_zones: Optional[List[UserEvacCenter]] = Field(None, description="List of user-defined evacuation zones with optional capacities.")
    buses_count: int = Field(3, description="Number of buses to use for homogeneous fleet.", gt=0)
    bus_capacity: int = Field(80, description="Capacity of each bus for homogeneous fleet.", gt=0)
    time_limit_seconds: int = Field(30, description="Maximum runtime for the algorithm in seconds.", gt=0)
    
    # NEW: Heterogeneous fleet definition
    vehicles: Optional[List[Vehicle]] = Field(None, description="List of custom vehicles. If provided, overrides buses_count and bus_capacity.")
    
    # EA specific parameters (optional for Pendelverkehr)
    population_size: Optional[int] = Field(200, description="Population size for the evolutionary algorithm.", gt=0)
    crossover_rate: Optional[float] = Field(0.8, description="Crossover rate for EA2.", ge=0, le=1)
    mutation_rate: Optional[float] = Field(0.2, description="Mutation rate for EA2.", ge=0, le=1)
    tournament_size: Optional[int] = Field(3, description="Tournament size for EA2 selection.", gt=1)
    
    # These are part of EA2 but I will keep them for now, the algorithm handles them.
    penalty_factor: float = Field(1000, description="Penalty for unserved demand.", ge=0)
    lateness_penalty_factor: float = Field(50, description="Penalty for late pickups.", ge=0)
    # NEW: custom center and buffer
    default_evac_center_coords: Optional[Tuple[float, float]] = Field(None, description="Coordinates for the main evacuation center [lon, lat]. If not provided, a default is used.")
    buffer_meters: Optional[int] = Field(None, description="Buffer in meters around the main center to select facilities. If not provided, a default is used.", gt=0)
    latest_evacuation_penalty_factor: float = Field(5.0, description="Penalty factor for the latest evacuation time.", ge=0)

    # NEW: Dynamic service time parameters
    use_dynamic_service_time: bool = Field(False, description="Use a dynamic service time based on the number of people per stop.")
    service_time_base_min: float = Field(3.0, description="Base service time in minutes per stop (e.g., for parking and setup).", ge=0)
    service_time_per_person_seconds: float = Field(20.0, description="Additional service time in seconds for each person picked up.", ge=0)
    alns_config: Optional[dict[str, Any]] = Field(None, description="Optional ALNS configuration overrides.")

    class Config:
        schema_extra = {
            "example": {
                "algorithm_choice": "ea2_memetic",
                "evacuation_zones": [
                    {
                        "label": "User-defined Center 1",
                        "coords": [10.051, 53.562],
                        "capacity": 500
                    }
                ],
                "buses_count": 3,
                "bus_capacity": 80,
                "time_limit_seconds": 60,
                "vehicles": [
                    {"capacity": 100, "start_depot": 0},
                    {"capacity": 50, "start_depot": 0},
                    {"capacity": 75, "start_node": 5, "start_coord": {"lat": 53.5, "lon": 10.1}}
                ],
                "population_size": 200,
                "crossover_rate": 0.85,
                "mutation_rate": 0.15,
                "tournament_size": 5,
                "penalty_factor": 1000,
                "lateness_penalty_factor": 50,
                "latest_evacuation_penalty_factor": 1.0,
                "default_evac_center_coords": [10.05, 53.56],
                "buffer_meters": 2000,
                "use_dynamic_service_time": True,
                "service_time_base_min": 3.0,
                "service_time_per_person_seconds": 20
            }
        }


class OptimizationJobCreateResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str


class OptimizationJobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


def _now_iso() -> str:
    return datetime.now().isoformat()


def _set_job_fields(job_id: str, **updates: Any) -> None:
    with _optimization_jobs_lock:
        job = _optimization_jobs.get(job_id)
        if job is None:
            return
        job.update(updates)


def _get_job_snapshot(job_id: str) -> dict[str, Any]:
    with _optimization_jobs_lock:
        job = _optimization_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Optimization job not found")
        return dict(job)


def _run_optimization_payload(payload: OptimizationPayload) -> dict[str, Any]:
    evacuation_zones_list = [zone.dict() for zone in payload.evacuation_zones] if payload.evacuation_zones else None

    # --- Pre-processing for custom coordinates ---
    start_to_node_seconds = {}
    if payload.vehicles and payload.algorithm_choice in ["ea2", "ea2_memetic", "pendelverkehr", "alns"]:
        try:
            current_center_coords = payload.default_evac_center_coords if payload.default_evac_center_coords else default_center_coords
            current_buffer_meters = payload.buffer_meters if payload.buffer_meters else default_buffer_meters
            buffer_poly = create_buffer_polygon(current_center_coords, current_buffer_meters)
            facility_features = load_facility_data(buffer_poly, current_buffer_meters)

            facility_coords = [feat["geometry"]["coordinates"] for feat in facility_features]

            if facility_coords:
                ors_client = ORSClient(key=ORS_KEY)
                for bus_idx, vehicle in enumerate(payload.vehicles):
                    if vehicle.start_coord:
                        print(f"🌐 Pre-calculating ORS route for vehicle {bus_idx} from custom coordinate...")
                        start_lon, start_lat = vehicle.start_coord.lon, vehicle.start_coord.lat

                        locations = [[start_lon, start_lat]] + facility_coords

                        matrix_result = ors_client.distance_matrix(
                            locations=locations,
                            sources=[0],
                            metrics=["duration"],
                        )

                        durations_to_facilities = matrix_result["durations"][0][1:]

                        start_to_node_seconds[bus_idx] = {
                            facility_idx: duration
                            for facility_idx, duration in enumerate(durations_to_facilities)
                        }
                        print(f"✅ Calculated {len(durations_to_facilities)} routes for vehicle {bus_idx}.")
        except Exception as e:
            print(f"⚠️  Warning: Could not pre-calculate ORS routes for custom coordinates: {e}")
            # Algorithm will fall back to Haversine distance estimation.

    vehicles_list = [v.dict(exclude_unset=True) for v in payload.vehicles] if payload.vehicles else None

    # Prepare dynamic service time parameters
    service_params = {
        "use_dynamic_service_time": payload.use_dynamic_service_time,
        "service_time_base_min": payload.service_time_base_min,
        "service_time_per_person_min": payload.service_time_per_person_seconds / 60.0,
    }

    if payload.algorithm_choice in ["ea2", "ea2_memetic"]:
        use_local_search = payload.algorithm_choice == "ea2_memetic"
        generations_limit = 10000

        result = run_evolutionary_algorithm(
            evacuation_zones_input=evacuation_zones_list,
            buses_count=payload.buses_count,
            bus_capacity=payload.bus_capacity,
            population_size=payload.population_size,
            generations=generations_limit,
            time_limit_seconds=payload.time_limit_seconds,
            use_local_search=use_local_search,
            crossover_rate=payload.crossover_rate,
            mutation_rate=payload.mutation_rate,
            tournament_size=payload.tournament_size,
            penalty_factor=payload.penalty_factor,
            lateness_penalty_factor=payload.lateness_penalty_factor,
            latest_evacuation_penalty_factor=payload.latest_evacuation_penalty_factor,
            default_evac_center_coords=payload.default_evac_center_coords,
            buffer_meters=payload.buffer_meters,
            vehicles=vehicles_list,
            start_to_node_seconds=start_to_node_seconds,
            **service_params,
        )

    elif payload.algorithm_choice == "pendelverkehr":
        pendel_algo = PendelverkehrShuttleAlgorithm()
        result = pendel_algo.run(
            evacuation_zones_input=evacuation_zones_list,
            buses_count=payload.buses_count,
            bus_capacity=payload.bus_capacity,
            vehicles=vehicles_list,
            default_evac_center_coords=payload.default_evac_center_coords,
            buffer_meters=payload.buffer_meters,
            start_to_node_seconds=start_to_node_seconds,
            **service_params,
        )

    elif payload.algorithm_choice == "alns":
        alns_algo = ALNSEvacuationAlgorithm()
        result = alns_algo.run(
            evacuation_zones_input=evacuation_zones_list,
            buses_count=payload.buses_count,
            bus_capacity=payload.bus_capacity,
            vehicles=vehicles_list,
            time_limit_seconds=payload.time_limit_seconds,
            default_evac_center_coords=payload.default_evac_center_coords,
            buffer_meters=payload.buffer_meters,
            start_to_node_seconds=start_to_node_seconds,
            alns_config=payload.alns_config,
            **service_params
        )

    else:
        raise HTTPException(status_code=400, detail=f"Invalid algorithm choice: {payload.algorithm_choice}")

    result["timestamp"] = datetime.now().isoformat()
    result["algorithm"] = payload.algorithm_choice
    return result


def _run_optimization_job(job_id: str, payload: OptimizationPayload) -> None:
    try:
        with _optimization_run_lock:
            _set_job_fields(job_id, status="running", started_at=_now_iso())
            result = _run_optimization_payload(payload)

        _set_job_fields(job_id, status="completed", finished_at=_now_iso(), result=result)
    except Exception as e:
        import traceback

        traceback.print_exc()
        _set_job_fields(job_id, status="failed", finished_at=_now_iso(), error=str(e))

# Allow CORS for frontend development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Bus-Based Evacuation Planning API"}


@app.get("/facilities")
def get_facilities(
    center_lon: Optional[float] = Query(None),
    center_lat: Optional[float] = Query(None),
    buffer_m: Optional[int] = Query(None)
):
    """
    Get the list of facilities without running optimization.
    This is a lightweight endpoint for the frontend's initial load.
    It can be customized with query parameters.
    """
    import json
    import os
    from shapely.geometry import Point
    
    # Use provided coordinates/buffer or fall back to defaults
    current_center_coords = (center_lon, center_lat) if center_lon is not None and center_lat is not None else default_center_coords
    current_buffer_meters = buffer_m if buffer_m is not None else default_buffer_meters
    
    # The buffer center is not a depot. Depots are managed by user input on the frontend.
    depots = []
    
    # Create buffer polygon dynamically
    buffer_poly = create_buffer_polygon(current_center_coords, current_buffer_meters)
    
    # Load and filter facilities (using core functions)
    filtered_features = load_facility_data(buffer_poly, current_buffer_meters)
    
    facilities = []
    for feature in filtered_features:
        props = feature["properties"]
        street = props.get("adresse", "Unknown Street")
        city = props.get("ort", "Unknown City")
        people_count = props.get("plaetze", 1)
        facility_address = f"{street}, {city}"
        lon, lat = feature["geometry"]["coordinates"]
        
        facilities.append({
            "label": facility_address,
            "coords": (lon, lat),
            "people": people_count
        })
    
    print(f"Returning {len(facilities)} facilities within {current_buffer_meters} m buffer")
    return {"depots": depots, "facilities": facilities}


@app.post("/optimization-jobs", response_model=OptimizationJobCreateResponse, summary="Start an optimization job (async)")
def start_optimization_job(payload: OptimizationPayload):
    job_id = str(uuid4())
    created_at = _now_iso()

    with _optimization_jobs_lock:
        _optimization_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": created_at,
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }

    Thread(target=_run_optimization_job, args=(job_id, payload), daemon=True).start()
    return {"job_id": job_id, "status": "queued", "created_at": created_at}


@app.get("/optimization-jobs/{job_id}", response_model=OptimizationJobStatusResponse, summary="Get optimization job status")
def get_optimization_job(job_id: str):
    return _get_job_snapshot(job_id)

@app.post("/run-optimization", summary="Run a selected evacuation optimization algorithm")
def run_optimization(background_tasks: BackgroundTasks, payload: OptimizationPayload):
    """
    Run an evacuation optimization algorithm based on the provided parameters.
    
    - **algorithm_choice**: Selects the algorithm to run ('ea2', 'ea2_memetic', 'pendelverkehr', 'alns').
    - **time_limit_seconds**: Sets the maximum runtime for the algorithm.
    - **vehicles**: An optional list to define a heterogeneous fleet, overriding `buses_count` and `bus_capacity`.
    - Other parameters are specific to the problem or the chosen algorithm.
    """
    evacuation_zones_list = [zone.dict() for zone in payload.evacuation_zones] if payload.evacuation_zones else None
    
    result = None
    
    # --- NEW: Pre-processing for custom coordinates ---
    start_to_node_seconds = {}
    # FIX: Include 'pendelverkehr' and 'alns' in the list of algorithms that trigger ORS requests.
    if payload.vehicles and payload.algorithm_choice in ["ea2", "ea2_memetic", "pendelverkehr", "alns"]:
        try:
            # Load facilities to get their coordinates for the ORS matrix call.
            current_center_coords = payload.default_evac_center_coords if payload.default_evac_center_coords else default_center_coords
            current_buffer_meters = payload.buffer_meters if payload.buffer_meters else default_buffer_meters
            buffer_poly = create_buffer_polygon(current_center_coords, current_buffer_meters)
            facility_features = load_facility_data(buffer_poly, current_buffer_meters)
            
            facility_coords = [feat["geometry"]["coordinates"] for feat in facility_features]

            if facility_coords:
                ors_client = ORSClient(key=ORS_KEY)
                for bus_idx, vehicle in enumerate(payload.vehicles):
                    if vehicle.start_coord:
                        print(f"🚀 Pre-calculating ORS route for vehicle {bus_idx} from custom coordinate...")
                        start_lon, start_lat = vehicle.start_coord.lon, vehicle.start_coord.lat
                        
                        locations = [[start_lon, start_lat]] + facility_coords
                        
                        matrix_result = ors_client.distance_matrix(
                            locations=locations,
                            sources=[0],  # Only calculate from our custom start point
                            metrics=["duration"]
                        )
                        
                        durations_to_facilities = matrix_result['durations'][0][1:]
                        
                        start_to_node_seconds[bus_idx] = {
                            facility_idx: duration
                            for facility_idx, duration in enumerate(durations_to_facilities)
                        }
                        print(f"✅ Calculated {len(durations_to_facilities)} routes for vehicle {bus_idx}.")
        except Exception as e:
            print(f"⚠️ Warning: Could not pre-calculate ORS routes for custom coordinates: {e}")
            # Algorithm will fall back to Haversine distance estimation.
            
    try:
        vehicles_list = [v.dict(exclude_unset=True) for v in payload.vehicles] if payload.vehicles else None

        # Prepare dynamic service time parameters
        service_params = {
            "use_dynamic_service_time": payload.use_dynamic_service_time,
            "service_time_base_min": payload.service_time_base_min,
            "service_time_per_person_min": payload.service_time_per_person_seconds / 60.0
        }

        if payload.algorithm_choice in ["ea2", "ea2_memetic"]:
            use_local_search = payload.algorithm_choice == "ea2_memetic"
            generations_limit = 10000 

            result = run_evolutionary_algorithm(
                evacuation_zones_input=evacuation_zones_list,
                buses_count=payload.buses_count,
                bus_capacity=payload.bus_capacity,
                population_size=payload.population_size,
                generations=generations_limit,
                time_limit_seconds=payload.time_limit_seconds,
                use_local_search=use_local_search,
                crossover_rate=payload.crossover_rate,
                mutation_rate=payload.mutation_rate,
                tournament_size=payload.tournament_size,
                penalty_factor=payload.penalty_factor,
                lateness_penalty_factor=payload.lateness_penalty_factor,
                latest_evacuation_penalty_factor=payload.latest_evacuation_penalty_factor,
                default_evac_center_coords=payload.default_evac_center_coords,
                buffer_meters=payload.buffer_meters,
                vehicles=vehicles_list,
                start_to_node_seconds=start_to_node_seconds,  # Pass pre-calculated data
                **service_params,  # Pass dynamic service time params
            )
        
        elif payload.algorithm_choice == "pendelverkehr":
            pendel_algo = PendelverkehrShuttleAlgorithm()
            result = pendel_algo.run(
                evacuation_zones_input=evacuation_zones_list,
                buses_count=payload.buses_count,
                bus_capacity=payload.bus_capacity,
                vehicles=vehicles_list,
                default_evac_center_coords=payload.default_evac_center_coords,
                buffer_meters=payload.buffer_meters,
                # FIX: Pass the pre-calculated ORS data to the baseline algorithm.
                start_to_node_seconds=start_to_node_seconds,
                **service_params, # Pass dynamic service time params
            )

        elif payload.algorithm_choice == "alns":
            alns_algo = ALNSEvacuationAlgorithm()
            result = alns_algo.run(
                evacuation_zones_input=evacuation_zones_list,
                buses_count=payload.buses_count,
                bus_capacity=payload.bus_capacity,
                vehicles=vehicles_list,
                time_limit_seconds=payload.time_limit_seconds,
                default_evac_center_coords=payload.default_evac_center_coords,
                buffer_meters=payload.buffer_meters,
                start_to_node_seconds=start_to_node_seconds,
                alns_config=payload.alns_config,
                **service_params
            )

        else:
            raise HTTPException(status_code=400, detail=f"Invalid algorithm choice: {payload.algorithm_choice}")

        result["timestamp"] = datetime.now().isoformat()
        result["algorithm"] = payload.algorithm_choice
        
        return result
        
    except Exception as e:
        print(f"Error running evacuation optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ors-proxy")
async def ors_proxy(request_body: dict):
    ors_url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
    headers = {
        "Authorization": ORS_KEY,
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(ors_url, json=request_body, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
