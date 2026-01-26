// Path: frontend2/src/app/data/scenarios.ts

/**
 * Preset scenario for a bomb evacuation.
 */
export const BOMB_EVACUATION_PRESET = {
  name: "Bomb Evacuation",
  mainCenter: [9.9592, 53.5592] as [number, number],      // lon, lat of the bomb
  bufferMeters: 300,
  evacCenter: { label: "Notunterkunft", coords: [9.9467, 53.5667] as [number, number], capacity: 500 }, // lon, lat of shelter
  vehicleStart: [9.9483, 53.5661] as [number, number],  // lon, lat for all vehicles
  fleet: [
    { capacity: 7, count: 2 },
    { capacity: 2, count: 4 },
    { capacity: 40, count: 1 },
  ],
};

/**
 * Preset scenario for a flood in Wilhelmsburg.
 */
export const WILHELMSBURG_FLOOD_PRESET = {
  name: "Überschwemmung Wilhelmsburg",
  mainCenter: [10.001287568640869, 53.507241291000895] as [number, number], // lon, lat
  bufferMeters: 3000,
  evacCenters: [
    { label: "Schule Mümmelmannsberg", coords: [10.148526582655167, 53.53143889821207] as [number, number], capacity: 200 },
    { label: "A-v-H-Gymnasium", coords: [9.993166088434934, 53.438736688521004] as [number, number], capacity: 200 },
    { label: "Schule Maretstraße", coords: [9.982006240324646, 53.4542489338715] as [number, number], capacity: 200 },
  ],
  vehicleStart: [10.149727113341193, 53.5231931772924] as [number, number], // lon, lat
  fleet: [
    { capacity: 2, count: 22 },
    { capacity: 7, count: 9 },
    { capacity: 40, count: 2 },
  ], 
};

/**
 * NEW: Default scenario for basic tests.
 */
export const DEFAULT_SCENARIO_PRESET = {
  name: "Default",
  mainCenter: [9.996754980861652, 53.49221335731889] as [number, number],
  bufferMeters: 5000,
  evacCenters: [
      { label: "Default Center", coords: [9.996754980861652, 53.49221335731889] as [number, number], capacity: 600 },
      { label: "Schule Mümmelmannsberg", coords: [10.148526582655167, 53.53143889821207] as [number, number], capacity: 200 },
      { label: "A-v-H-Gymnasium", coords: [9.993166088434934, 53.438736688521004] as [number, number], capacity: 200 },
      { label: "Schule Maretstraße", coords: [9.982006240324646, 53.4542489338715] as [number, number], capacity: 200 },
      { label: "Notunterkunft", coords: [9.9467, 53.5667] as [number, number], capacity: 500 },
  ],
  vehicleStart: null, // Use depot as start
  fleet: [
      { capacity: 80, count: 3 }
  ]
};