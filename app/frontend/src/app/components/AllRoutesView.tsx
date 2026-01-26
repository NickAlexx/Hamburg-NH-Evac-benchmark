// Path: frontend2/src/app/components/AllRoutesView.tsx
'use client';

import React from 'react';
import { Trip, BackendVehicle } from './types'; // Import BackendVehicle

interface Location {
  label: string;
  coords: [number, number]; // (lon, lat)
  people: number;
}

interface TripSimulation {
  details: string[];
  departure: number;
  return: number;
  trip_time: number;
}

interface SimulationData {
  [busIndex: string]: {
    [tripIndex: string]: TripSimulation;
  };
}

interface AllRoutesViewProps {
  simulationData: SimulationData;
  bestSolution: Trip[][];
  locationData: Location[];
  numDepots: number;
  onClose: () => void;
  vehicles?: BackendVehicle[]; // Add this prop
}

const AllRoutesView: React.FC<AllRoutesViewProps> = ({
  simulationData,
  bestSolution,
  locationData,
  numDepots,
  onClose,
  vehicles,
}) => {
  // Helper to get formatted details for a single stop in a compact string
  const getStopDetailsString = (
    facilityIndex: number,
    stopIndexInTrip: number, // The new, crucial parameter: 0 for 1st stop, 1 for 2nd...
    stopActionDetails: string[] // Pre-filtered list of "Stop ..." lines
  ): string => {
    const facilityLabel = locationData[numDepots + facilityIndex]?.label?.split(',')[0] || `Facility ${facilityIndex}`;
    
    // Use the index to get the correct detail line, this is much more reliable
    const detailLine = stopActionDetails[stopIndexInTrip];
    let info = "pass-through";

    if (detailLine) {
      const pickupMatch = detailLine.match(/picked up (\d+)/);
      const timeMatch = detailLine.match(/at time ([\d.]+) min/);
      const isLate = detailLine.includes("(late)");
      
      if (detailLine.includes("no evacuees")) {
        info = "no pickup";
      } else if (pickupMatch) {
        const pickupCount = parseInt(pickupMatch[1], 10);
        if (pickupCount > 0) {
          const timeInfo = timeMatch ? `@ ${parseFloat(timeMatch[1]).toFixed(1)}min` : '';
          const lateInfo = isLate ? ' (LATE)' : '';
          info = `picked up ${pickupCount} ${timeInfo}${lateInfo}`;
        } else {
            info = "no pickup"; // Handle case where "picked up 0" is present
        }
      }
    }
    return `${facilityLabel} [${info}]`;
  };

  // Helper to describe the vehicle's start origin
  const getStartOriginString = (vehicle?: BackendVehicle): string => {
    if (!vehicle || !vehicle.start) {
      return 'Default (Depot 0)';
    }
    const { kind, index, lat, lon } = vehicle.start;
    if (kind === 'depot') {
      const depotLabel = locationData[index ?? 0]?.label.split(',')[0] || `Depot ${index ?? 0}`;
      return `Depot: ${depotLabel}`;
    }
    if (kind === 'node') {
      const facilityLabel = locationData[numDepots + (index ?? 0)]?.label.split(',')[0] || `Facility ${index ?? 0}`;
      return `Facility: ${facilityLabel}`;
    }
    if (kind === 'coord' && lat !== undefined && lon !== undefined) {
      return `Custom Coordinate (${lat.toFixed(3)}, ${lon.toFixed(3)})`;
    }
    return 'Default (Depot 0)';
  };

  return (
    <div className="all-routes-modal-overlay">
      <div className="all-routes-modal-content">
        <button className="all-routes-modal-close" onClick={onClose}>
          ×
        </button>
        <h2>All Routes</h2>
        
        {bestSolution.map((busSchedule, busIdx) => {
          const vehicle = vehicles?.[busIdx];
          const capacityInfo = vehicle ? ` (Capacity: ${vehicle.capacity})` : '';
          const startOriginInfo = getStartOriginString(vehicle);

          return (
            <div key={`bus-summary-${busIdx}`} className="bus-route-summary">
              <h3>Vehicle {busIdx}{capacityInfo}</h3>
              <p style={{ marginTop: '-10px', marginBottom: '15px', fontSize: '0.9rem', color: '#555' }}>
                <strong>Start Origin:</strong> {startOriginInfo}
              </p>
              
              {busSchedule.length > 0 ? (
                <ul>
                  {busSchedule.map((trip, tripIdx) => {
                    const tripSim = simulationData[String(busIdx)]?.[String(tripIdx)];
                    const tripSimDetails = tripSim?.details || [];

                    // *** FIX: Pre-filter the detail lines that correspond to stop actions ***
                    const stopActionDetails = tripSimDetails.filter(line => 
                        line.startsWith('Stop ') && (line.includes('picked up') || line.includes('no evacuees'))
                    );

                    let startDepotLabel = locationData[trip.start_depot]?.label.split(',')[0] || `Depot ${trip.start_depot}`;
                    if (tripIdx === 0 && vehicle?.start?.kind === 'coord') {
                      startDepotLabel = 'Custom Coordinate Start';
                    }
                    const endDepotLabel = locationData[trip.end_depot]?.label.split(',')[0] || `Depot ${trip.end_depot}`;

                    const facilityStops = trip.stops
                      // Pass the stop's index within the trip (s_idx) to the helper function
                      .map((stopIdx, s_idx) => getStopDetailsString(stopIdx, s_idx, stopActionDetails))
                      .join(' → ');
                    
                    const fullRoute = facilityStops.length > 0 
                      ? `${startDepotLabel} → ${facilityStops} → ${endDepotLabel}`
                      : `${startDepotLabel} → ${endDepotLabel}`;
                    
                    return (
                      <li key={`trip-summary-${busIdx}-${tripIdx}`}>
                        <strong>Trip {tripIdx}:</strong> 
                        <span style={{color: '#555', fontSize: '0.9em', marginLeft: '5px'}}>
                          (Dep: {tripSim?.departure.toFixed(0)} min, Ret: {tripSim?.return.toFixed(0)} min)
                        </span>
                        <div style={{ paddingLeft: '15px', marginTop: '4px', lineHeight: '1.4' }}>
                          {fullRoute}
                        </div>
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <p className="no-trips">No trips assigned to this vehicle.</p>
              )}
            </div>
          )
        })}
      </div>
    </div>
  );
};

export default AllRoutesView;
