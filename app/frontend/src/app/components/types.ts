// Path: frontend2/src/app/components/types.ts

export interface Trip {
  start_depot: number;
  stops: number[];
  end_depot: number;
  pickup_counts: { [key: number]: number };
}

export interface BusPosition {
  busId: number;
  position: [number, number]; // [lat, lng]
  isMoving: boolean;
  currentTrip?: string;
  nextStop?: string;
}

export interface FacilityStatus {
  facilityId: number;
  initialEvacuees: number;
  remainingEvacuees: number;
}

export interface EvacuationMetrics {
  averageEvacTime: number;
  latestEvacTime: number;
  cumulativeWaitingTime: number;
  totalEvacuees: number;
}

export interface UserEvacCenter {
  label: string;
  coords: [number, number];
}

// Frontend-specific vehicle type for the editor
export interface Vehicle {
  id: string;
  capacity: number;
  start: 
    | { kind: 'depot', index: number }
    | { kind: 'node', index: number }
    | { kind: 'coord', coords: [number, number] | null }; // [lon, lat] or null if not set
}

// Type for the vehicle data received from the backend
export interface BackendVehicle {
  id?: string | null;
  capacity: number;
  start: {
    kind: 'depot' | 'node' | 'coord';
    index?: number;
    lat?: number;
    lon?: number;
  };
}

// Type for the payload sent to the backend
export interface BackendVehiclePayload {
    capacity: number;
    start_depot?: number;
    start_node?: number;
    start_coord?: { lat: number; lon: number };
}