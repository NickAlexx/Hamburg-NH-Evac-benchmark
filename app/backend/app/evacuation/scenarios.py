# path: backend/app/evacuation/scenarios.py

"""
Defines the expert-validated evacuation scenarios for experimental evaluation.
This serves as the single source of truth for the paper's case studies.
"""

# ==============================================================================
# CASE STUDY 1: BOMB THREAT EVACUATION (BAHRENFELD)
# ==============================================================================

BOMB_THREAT_SCENARIO = {
    "name": "BombThreat",
    "description": "Evacuation of a 400m radius due to a 1000kg WWII bomb.",
    "main_center": [9.9592, 53.5592],  # lon, lat
    "buffer_meters": 300,
    "evac_centers": [
        # TS `evacCenter` becomes a list of one
        {"label": "Notunterkunft", "coords": [9.9467, 53.5667], "capacity": 500}  # lon, lat
    ],
    "vehicle_start_point": [9.9483, 53.5661],  # lon, lat
    "fleets": {
        "specialized_only": [
            {"type": "N-KTW", "capacity": 2, "count": 4},
            {"type": "MZF", "capacity": 7, "count": 2},
        ],
        "augmented": [
            {"type": "N-KTW", "capacity": 2, "count": 4},
            {"type": "MZF", "capacity": 7, "count": 2},
            {"type": "Bus", "capacity": 40, "count": 1},
        ]
    }
}

# ==============================================================================
# CASE STUDY 2: FLOOD EVACUATION (WILHELMSBURG)
# ==============================================================================

FLOOD_SCENARIO = {
    "name": "FloodWilhelmsburg",
    "description": "Evacuation of a 3000m radius in Wilhelmsburg due to a storm surge.",
    "main_center": [10.001287568640869, 53.507241291000895],  # lon, lat
    "buffer_meters": 3000,
    "evac_centers": [
        {"label": "Schule Mümmelmannsberg", "coords": [10.148526582655167, 53.53143889821207], "capacity": 200},
        {"label": "A-v-H-Gymnasium", "coords": [9.993166088434934, 53.438736688521004], "capacity": 200},
        {"label": "Schule Maretstraße", "coords": [9.982006240324646, 53.4542489338715], "capacity": 200},
    ],
    "vehicle_start_point": [10.149727113341193, 53.5231931772924],  # lon, lat
    "fleets": {
        "specialized_only": [
            {"type": "N-KTW", "capacity": 2, "count": 22},
            {"type": "MZF", "capacity": 7, "count": 9},
        ],
        "augmented": [
            {"type": "N-KTW", "capacity": 2, "count": 22},
            {"type": "MZF", "capacity": 7, "count": 9},
            {"type": "Bus", "capacity": 40, "count": 2},
        ]
    }
}

# ==============================================================================
# DEFAULT SCENARIO (for basic tests)
# ==============================================================================

DEFAULT_SCENARIO = {
    "name": "Default",
    "description": "Default scenario with 3 buses, each with 80 capacity.",
    "main_center": [9.996754980861652, 53.49221335731889],
    "buffer_meters": 5000,
    "evac_centers": [
        {"label": "Default Center", "coords": [9.996754980861652, 53.49221335731889], "capacity": 600},
        {"label": "Schule Mümmelmannsberg", "coords": [10.148526582655167, 53.53143889821207], "capacity": 200},
        {"label": "A-v-H-Gymnasium", "coords": [9.993166088434934, 53.438736688521004], "capacity": 200},
        {"label": "Schule Maretstraße", "coords": [9.982006240324646, 53.4542489338715], "capacity": 200},
        {"label": "Notunterkunft", "coords": [9.9467, 53.5667], "capacity": 500},
    ],
    "vehicle_start_point": None,  # Use depot as start
    "fleets": {
        "default": [
            {"type": "Bus", "capacity": 80, "count": 3}
        ]
    }
}


# Dictionary to easily access all scenarios
ALL_SCENARIOS = {
    "bomb": BOMB_THREAT_SCENARIO,
    "flood": FLOOD_SCENARIO,
    "default": DEFAULT_SCENARIO
}