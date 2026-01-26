// Path: frontend2/src/app/components/BusGanttChart.tsx
'use client';

import React, { useEffect, useState, useRef } from "react";
import { BackendVehicle, Trip } from "./types";

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

interface Location {
  label: string;
  coords: [number, number];
  people: number;
}

interface BusGanttChartProps {
  simulationData: SimulationData;
  maxTime?: number;
  maxHeight?: number;
  vehicles?: BackendVehicle[]; // Add this prop
  bestSolution: Trip[][];
  locationData: Location[];
  numDepots: number;
}

interface TripData {
  busId: string;
  tripId: string;
  start: number;
  end: number;
  label: string;
  color: string;
  details: string[];
  hasLatePicups: boolean;
}

const BusGanttChart: React.FC<BusGanttChartProps> = ({
  simulationData,
  maxTime,
  maxHeight = 200,
  vehicles,
  bestSolution,
  locationData,
  numDepots,
}) => {
  const [trips, setTrips] = useState<TripData[]>([]);
  const [chartMaxTime, setChartMaxTime] = useState<number>(maxTime || 180);
  const [hoveredTripInfo, setHoveredTripInfo] = useState<string | null>(null);
  const chartRef = useRef<HTMLDivElement>(null);
  
  // Bus colors - one color per bus for consistency
  const busColors = [
    '#3498db', // blue
    '#2ecc71', // green
    '#e74c3c', // red
    '#f39c12', // orange
    '#9b59b6', // purple
    '#1abc9c', // turquoise
    '#d35400', // dark orange
    '#2c3e50', // dark blue
    '#27ae60', // dark green
    '#c0392b'  // dark red
  ];

  // Process data when simulation data changes
  useEffect(() => {
    if (Object.keys(simulationData).length === 0) return;

    const newTrips: TripData[] = [];
    let calculatedMaxTime = maxTime || 0;
    
    // Sort bus indices numerically
    const busIndices = Object.keys(simulationData).sort((a, b) => 
      parseInt(a, 10) - parseInt(b, 10)
    );
    
    busIndices.forEach(busIdx => {
      const busTrips = simulationData[busIdx];
      const busColor = busColors[parseInt(busIdx, 10) % busColors.length];
      
      // Sort trip indices numerically
      const tripIndices = Object.keys(busTrips).sort((a, b) => 
        parseInt(a, 10) - parseInt(b, 10)
      );
      
      tripIndices.forEach(tripIdx => {
        const trip = busTrips[tripIdx];
        
        // Update max time if needed
        if (trip.return > calculatedMaxTime) {
          calculatedMaxTime = trip.return;
        }
        
        // Check if the trip has any late pickups
        const hasLatePicups = trip.details.some(detail => 
          detail.includes("(late)") && detail.includes("picked up")
        );
        
        // Create trip data object
        newTrips.push({
          busId: busIdx,
          tripId: tripIdx,
          start: trip.departure,
          end: trip.return,
          label: `Trip ${tripIdx}`,
          color: busColor,
          details: trip.details,
          hasLatePicups
        });
      });
    });
    
    setTrips(newTrips);
    setChartMaxTime(calculatedMaxTime + 10); // Add padding
  }, [simulationData, maxTime]);

  // Calculate trip statistics for tooltip
  const getTripStats = (trip: TripData): string => {
    let totalEvacuees = 0;
    const facilityNames: string[] = [];
    const latePickups = trip.details.filter(detail => 
      detail.includes("(late)") && detail.includes("picked up")
    ).length;
    
    trip.details.forEach(detail => {
      const pickupMatch = detail.match(/picked up (\d+)/);
      if (pickupMatch) {
        totalEvacuees += parseInt(pickupMatch[1], 10);
      }
      
      // Extract facility names
      if (detail.startsWith('Stop ') && detail.includes('picked up')) {
        const facilityName = detail.split(':')[0].replace('Stop ', '').trim();
        const isLate = detail.includes('(late)');
        if (isLate) {
          facilityNames.push(`${facilityName} ⚠️ (late)`);
        } else {
          facilityNames.push(facilityName);
        }
      }
    });

    const busIdx = parseInt(trip.busId, 10);
    const tripIdx = parseInt(trip.tripId, 10);
    const vehicle = vehicles?.[busIdx];
    const capacityInfo = vehicle ? ` (Cap: ${vehicle.capacity})` : '';

    // *** MODIFIED PART ***
    // Get start origin info for the tooltip
    let startOriginInfo = 'Default Start';
    if (vehicle && vehicle.start) {
      if (vehicle.start.kind === 'depot') {
        startOriginInfo = `Starts at Depot ${vehicle.start.index}`;
      } else if (vehicle.start.kind === 'node') {
        startOriginInfo = `Starts at Facility ${vehicle.start.index}`;
      } else if (vehicle.start.kind === 'coord') {
        startOriginInfo = 'Starts at Custom Coordinate';
      }
    }

    // Get depot information for the specific trip
    const tripObject = bestSolution?.[busIdx]?.[tripIdx];
    let routeInfo = '';
    if (tripObject && locationData) {
      let startDepotLabel = locationData[tripObject.start_depot]?.label.split(',')[0] || `Depot ${tripObject.start_depot}`;
      // If it's the first trip and the vehicle has a custom coordinate start, override the start depot label.
      if (tripIdx === 0 && vehicle?.start?.kind === 'coord') {
        startDepotLabel = 'Custom Start';
      }
      const endDepotLabel = locationData[tripObject.end_depot]?.label.split(',')[0] || `Depot ${tripObject.end_depot}`;
      routeInfo = `Route: ${startDepotLabel} → ${endDepotLabel}<br>`;
    }
    
    return `
      <strong>Vehicle ${trip.busId}${capacityInfo}</strong> | <em style="color:#555">${startOriginInfo}</em><br>
      <strong>Trip ${trip.tripId}</strong><br>
      Time: ${trip.start.toFixed(0)} - ${trip.end.toFixed(0)} min<br>
      Duration: ${(trip.end - trip.start).toFixed(1)} min<br>
      ${routeInfo}
      Evacuees: ${totalEvacuees}<br>
      ${latePickups > 0 ? `<span style="color:#e74c3c">Late pickups: ${latePickups}</span><br>` : ''}
      Stops: ${facilityNames.length > 0 ? ' • ' + facilityNames.join('<br> • ') : 'None'}
    `;
  };

  // Handle mouse enter/leave for tooltips
  const handleMouseEnter = (trip: TripData) => {
    setHoveredTripInfo(getTripStats(trip));
  };
  
  const handleMouseLeave = () => {
    setHoveredTripInfo(null);
  };

  // Sort unique bus IDs
  const uniqueBusIds = Array.from(new Set(trips.map(t => t.busId)))
    .sort((a, b) => parseInt(a) - parseInt(b));
    
  // Render warning label separately - don't attach it to the bar
  const renderWarningLabels = () => {
    return trips.filter(trip => trip.hasLatePicups).map(trip => {
      const busIdx = uniqueBusIds.indexOf(trip.busId);
      if (busIdx === -1) return null;
      
      // Calculate position
      const left = `${(trip.end / chartMaxTime) * 100}%`;
      
      return (
        <div
          key={`warning-${trip.busId}-${trip.tripId}`}
          className="warning-label"
          style={{
            position: 'absolute',
            left: left,
            top: `${busIdx * 30}px`, // Align with the top of the row
            height: '75px', // Match the height of gantt row
            display: 'flex',
            alignItems: 'center', // Center vertically
            marginLeft: '-20px', // Pull left to overlap end of bar
            fontSize: '14px',
            zIndex: 2,
            pointerEvents: 'none' // So it doesn't interfere with mouse events
          }}
          title="Contains late pickups"
        >
          ⚠️
        </div>
      );
    });
  };

  return (
    <div className="bus-gantt-chart" ref={chartRef}>
      <h3>Vehicle Timelines</h3>
      
      <div className="gantt-chart-container">
        {/* Y-axis labels (bus IDs) */}
        <div className="chart-labels">
          {uniqueBusIds.map(busId => {
            const busIdx = parseInt(busId, 10);
            const vehicle = vehicles?.[busIdx];
            const capacityInfo = vehicle ? `(C:${vehicle.capacity})` : '';
            return (
              <div key={`label-${busId}`} className="chart-label">
                Vehicle {busId} <small style={{fontWeight: 'normal', color: '#666'}}>{capacityInfo}</small>
              </div>
            );
          })}
        </div>
        
        {/* Chart area */}
        <div className="chart-area" style={{ position: 'relative' }}>
          {/* Time markers */}
          <div className="time-markers">
            {Array.from({ length: Math.ceil(chartMaxTime / 30) + 1 }, (_, i) => i * 30).map(time => (
              <div 
                key={`marker-${time}`} 
                className="time-marker" 
                style={{ left: `${(time / chartMaxTime) * 100}%` }}
              >
                {time}
              </div>
            ))}
          </div>
          
          {/* Bus rows */}
          {uniqueBusIds.map(busId => (
            <div key={`row-${busId}`} className="gantt-row">
              {trips.filter(t => t.busId === busId).map(trip => (
                <div 
                  key={`trip-${busId}-${trip.tripId}`}
                  className="gantt-bar"
                  style={{
                    left: `${(trip.start / chartMaxTime) * 100}%`,
                    width: `${((trip.end - trip.start) / chartMaxTime) * 100}%`,
                    backgroundColor: trip.color
                  }}
                  onMouseEnter={() => handleMouseEnter(trip)}
                  onMouseLeave={handleMouseLeave}
                >
                  {trip.end - trip.start > 15 ? trip.label : ''}
                </div>
              ))}
            </div>
          ))}
          
          {/* Warning icons - rendered separately */}
          {renderWarningLabels()}
        </div>
      </div>
      
      {/* Tooltip */}
      {hoveredTripInfo && (
        <div className="gantt-tooltip" dangerouslySetInnerHTML={{ __html: hoveredTripInfo }}></div>
      )}
    </div>
  );
};

export default BusGanttChart;
