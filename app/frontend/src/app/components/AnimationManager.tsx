'use client';

import React, { useState, useEffect, useRef } from 'react';
import TimelineController from './TimelineController';
import { Trip, BusPosition, FacilityStatus } from './types';

interface AnimationManagerProps {
  simulationData: SimulationData;
  bestSolution: Trip[][];
  routePaths: { [key: string]: [number, number][] };
  numDepots: number;
  locationData: Location[];
  onBusPositionsUpdate: (positions: BusPosition[]) => void;
  onFacilityStatusUpdate: (statuses: FacilityStatus[]) => void;
}

interface SimulationData {
  [busIndex: string]: {
    [tripIndex: string]: TripSimulation;
  };
}

interface TripSimulation {
  details: string[];
  departure: number;
  return: number;
  trip_time: number;
}

interface Location {
  label: string;
  coords: [number, number]; // (lon, lat)
  people: number;
}

const AnimationManager: React.FC<AnimationManagerProps> = ({
  simulationData,
  bestSolution,
  routePaths,
  numDepots,
  locationData,
  onBusPositionsUpdate,
  onFacilityStatusUpdate
}) => {
  // Timeline state
  const [startTime] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  
  // Animation frame tracking using refs to avoid re-renders
  const rafId = useRef<number | null>(null);
  const prevTimestampRef = useRef<number | null>(null);
  
  // Calculate the max simulation time (end time)
  useEffect(() => {
    let maxTime = 0;
    Object.keys(simulationData).forEach(busIdx => {
      const trips = simulationData[busIdx];
      Object.keys(trips).forEach(tripIdx => {
        const returnTime = trips[tripIdx].return;
        if (returnTime > maxTime) {
          maxTime = returnTime;
        }
      });
    });
    
    // Add a small buffer
    maxTime = Math.ceil(maxTime) + 5;
    setEndTime(maxTime);
  }, [simulationData]);

  // Initialize facility evacuation data
  const initializeFacilityDemand = () => {
    const facilityDemand: { [facilityId: number]: number } = {};
    // Start by getting the initial people count for all facilities
    for (let i = 0; i < locationData.length - numDepots; i++) {
      const facilityIdx = numDepots + i;
      if (facilityIdx < locationData.length) {
        facilityDemand[i] = locationData[facilityIdx].people;
      }
    }
    return facilityDemand;
  };

  // Function to calculate facility statuses at a given time
  const calculateFacilityStatus = (time: number): FacilityStatus[] => {
    // Initialize with the full demand for each facility
    const remainingDemand = initializeFacilityDemand();
    const statuses: FacilityStatus[] = [];
    
    // Process all trips up to the current time
    Object.keys(simulationData).forEach(busIdx => {
      const trips = simulationData[busIdx];
      const busSchedule = bestSolution[parseInt(busIdx, 10)] || [];
      
      // Sort trip indices numerically to ensure correct order
      const sortedTripIndices = Object.keys(trips).sort((a, b) => 
        parseInt(a, 10) - parseInt(b, 10)
      );
      
      for (const tripIdx of sortedTripIndices) {
        const trip = trips[tripIdx];
        const tripObject = busSchedule[parseInt(tripIdx, 10)];
        
        if (!tripObject || tripObject.stops.length === 0) continue;
        
        // Completed trips - process all pickups
        if (trip.return <= time) {
          processPickupsFromTripDetails(trip.details, tripObject.stops, remainingDemand);
          continue;
        }
        
        // For in-progress trips, calculate facility arrival times and process pickups
        if (trip.departure <= time && time < trip.return) {
          const routeKey = `${busIdx}-${tripIdx}`;
          const routePath = routePaths[routeKey];
          
          // Calculate arrival times for each facility in this trip
          const arrivalTimes = calculateFacilityArrivalTimes(trip, tripObject, routePath);
          
          // Process pickups for facilities where we've already arrived
          for (const facilityInfo of arrivalTimes) {
            if (facilityInfo.arrivalTime <= time) {
              processPickupForFacility(
                facilityInfo.facilityIndex,
                trip.details,
                remainingDemand
              );
            }
          }
        }
      }
    });
    
    // Convert to array of FacilityStatus objects
    for (const facilityId in remainingDemand) {
      const id = parseInt(facilityId, 10);
      const initialEvacuees = locationData[numDepots + id]?.people || 0;
      statuses.push({
        facilityId: id,
        initialEvacuees,
        remainingEvacuees: remainingDemand[id]
      });
    }
    
    return statuses;
  };

  // Function to calculate arrival times at each facility in a trip based on route path
  const calculateFacilityArrivalTimes = (
    trip: TripSimulation, 
    tripObject: Trip, 
    routePath?: [number, number][]
  ) => {
    const result: { facilityIndex: number; arrivalTime: number }[] = [];
    const departureTime = trip.departure;
    const totalTripTime = trip.trip_time;
    
    // If no stops or no path, return empty result
    if (!tripObject.stops || tripObject.stops.length === 0) {
      return result;
    }
    
    const numFacilities = tripObject.stops.length;
    
    // If we have route path coordinates, use them to estimate arrival times more accurately
    if (routePath && routePath.length > numFacilities) {
      // Estimate times based on route path segments
      
      // Calculate the total path length
      let totalPathLength = 0;
      for (let i = 0; i < routePath.length - 1; i++) {
        const [lon1, lat1] = routePath[i];
        const [lon2, lat2] = routePath[i + 1];
        // Simple distance calculation (could be improved with haversine formula)
        const segmentLength = Math.sqrt(
          Math.pow(lon2 - lon1, 2) + Math.pow(lat2 - lat1, 2)
        );
        totalPathLength += segmentLength;
      }
      
      // Map each facility to the closest point in the route path
      // This is a simplified approach - in a real implementation, you'd map 
      // each facility to its exact position in the path
      for (let i = 0; i < numFacilities; i++) {
        const facilityIdx = tripObject.stops[i];
        const facilityCoords = locationData[numDepots + facilityIdx]?.coords;
        
        if (!facilityCoords) continue;
        
        // Find the closest point in route path
        let closestPointIdx = 0;
        let minDistance = Infinity;
        
        for (let j = 0; j < routePath.length; j++) {
          const [pathLon, pathLat] = routePath[j];
          const [facLon, facLat] = facilityCoords;
          
          const distance = Math.sqrt(
            Math.pow(pathLon - facLon, 2) + Math.pow(pathLat - facLat, 2)
          );
          
          if (distance < minDistance) {
            minDistance = distance;
            closestPointIdx = j;
          }
        }
        
        // Calculate distance up to this point in the route
        let distanceToFacility = 0;
        for (let j = 0; j < closestPointIdx; j++) {
          const [lon1, lat1] = routePath[j];
          const [lon2, lat2] = routePath[j + 1];
          const segmentLength = Math.sqrt(
            Math.pow(lon2 - lon1, 2) + Math.pow(lat2 - lat1, 2)
          );
          distanceToFacility += segmentLength;
        }
        
        // Calculate proportional time to reach this facility
        const proportionalTime = totalPathLength > 0 
          ? departureTime + (distanceToFacility / totalPathLength) * totalTripTime
          : departureTime;
        
        result.push({
          facilityIndex: facilityIdx,
          arrivalTime: proportionalTime
        });
      }
      
      // Sort by arrival time
      result.sort((a, b) => a.arrivalTime - b.arrivalTime);
    } else {
      // Fallback to a simple proportional time allocation
      // This assumes stops are evenly spaced in time
      for (let i = 0; i < numFacilities; i++) {
        // Use a simple proportion: time = departure + (i+1)/(numFacilities+1) * totalTime
        // This divides the trip time into equal segments
        const proportionalTime = departureTime + ((i + 1) / (numFacilities + 1)) * totalTripTime;
        result.push({
          facilityIndex: tripObject.stops[i],
          arrivalTime: proportionalTime
        });
      }
    }
    
    return result;
  };
  
  // Helper function to process pickups from trip details
  const processPickupsFromTripDetails = (
    details: string[], 
    stops: number[], 
    remainingDemand: { [facilityId: number]: number }
  ) => {
    for (const detail of details) {
      // Look for patterns like "Stop [name]: picked up [number]"
      const pickupMatch = detail.match(/Stop .+: picked up (\d+)/);
      if (pickupMatch) {
        const amount = parseInt(pickupMatch[1], 10);
        
        // Try to determine which facility this is
        for (const stop of stops) {
          if (detail.includes(locationData[numDepots + stop]?.label)) {
            // Reduce the demand for this facility
            if (remainingDemand[stop] !== undefined) {
              remainingDemand[stop] = Math.max(0, remainingDemand[stop] - amount);
            }
            break;
          }
        }
      }
    }
  };
  
  // Helper to process a pickup for a specific facility
  const processPickupForFacility = (
    facilityId: number, 
    details: string[], 
    remainingDemand: { [facilityId: number]: number }
  ) => {
    // Find the matching facility name in the details
    const facilityLabel = locationData[numDepots + facilityId]?.label;
    if (!facilityLabel) return;
    
    for (const detail of details) {
      if (detail.includes(facilityLabel) && detail.includes("picked up")) {
        const pickupMatch = detail.match(/picked up (\d+)/);
        if (pickupMatch) {
          const amount = parseInt(pickupMatch[1], 10);
          if (remainingDemand[facilityId] !== undefined) {
            remainingDemand[facilityId] = Math.max(0, remainingDemand[facilityId] - amount);
            break;
          }
        }
      }
    }
  };

  // Function to calculate bus positions at a given time
  const calculateBusPositions = (time: number): BusPosition[] => {
    const positions: BusPosition[] = [];
    
    Object.keys(simulationData).forEach(busIdx => {
      const busIdNumber = parseInt(busIdx, 10);
      const trips = simulationData[busIdx];
      const busSchedule = bestSolution[busIdNumber] || [];
      
      // Find the current trip for this bus at this time
      let currentTripIdx: string | null = null;
      let nextStop: string | undefined;
      let busIsMoving = false;
      
      // Sort trip indices numerically to ensure correct order
      const sortedTripIndices = Object.keys(trips).sort((a, b) => 
        parseInt(a, 10) - parseInt(b, 10)
      );
      
      for (const tripIdx of sortedTripIndices) {
        const trip = trips[tripIdx];
        // If current time is during this trip
        if (time >= trip.departure && time <= trip.return) {
          currentTripIdx = tripIdx;
          busIsMoving = true;
          break;
        }
        // If current time is after this trip's return but before the next trip's departure
        const nextTripIdx = sortedTripIndices[sortedTripIndices.indexOf(tripIdx) + 1];
        if (nextTripIdx && time > trip.return && time < trips[nextTripIdx].departure) {
          // Bus is idle between trips
          currentTripIdx = null;
          busIsMoving = false;
          break;
        }
      }
      
      // Calculate position based on current trip
      let position: [number, number];
      
      if (currentTripIdx !== null && busIsMoving) {
        const tripDetails = trips[currentTripIdx];
        const tripIdx = parseInt(currentTripIdx, 10);
        const tripObject = busSchedule[tripIdx];
        
        if (!tripObject) {
          // Fallback to the depot position if trip object doesn't exist
          const depotIdx = 0; // Default to first depot
          const depotCoords = locationData[depotIdx]?.coords || [0, 0];
          position = [depotCoords[1], depotCoords[0]]; // Convert from [lon, lat] to [lat, lng]
        } else {
          // Get the route path for this trip
          const routeKey = `${busIdx}-${currentTripIdx}`;
          const path = routePaths[routeKey];
          
          if (path && path.length > 1) {
            // Calculate progress along the route based on time
            const tripProgress = (time - tripDetails.departure) / tripDetails.trip_time;
            
            // Use more precise interpolation for better position tracking
            const exactPathIndex = tripProgress * (path.length - 1);
            const pathIndex = Math.min(
              Math.floor(exactPathIndex),
              path.length - 2
            );
            
            // Get the decimal part for precise interpolation
            const subProgress = exactPathIndex - pathIndex;
            const p1 = path[pathIndex];
            const p2 = path[pathIndex + 1];
            
            // Linear interpolation between points
            const lat = p1[1] + (p2[1] - p1[1]) * subProgress;
            const lng = p1[0] + (p2[0] - p1[0]) * subProgress;
            
            position = [lat, lng];
            
            // Determine next stop
            if (tripObject.stops.length > 0) {
              const stopsProgress = tripProgress * tripObject.stops.length;
              const completedStops = Math.floor(stopsProgress);
              const nextStopIdx = Math.min(
                completedStops,
                tripObject.stops.length - 1
              );
              
              // If we've completed all stops, the next stop is the end depot
              if (completedStops >= tripObject.stops.length) {
                const endDepotIdx = tripObject.end_depot;
                nextStop = locationData[endDepotIdx]?.label || "End Depot";
              } else {
                // Otherwise, the next stop is the next facility
                const facilityIdx = numDepots + tripObject.stops[nextStopIdx];
                nextStop = locationData[facilityIdx]?.label;
              }
            }
          } else {
            // Fallback if path not available
            const depotIdx = tripObject.start_depot;
            const depotCoords = locationData[depotIdx]?.coords || [0, 0];
            position = [depotCoords[1], depotCoords[0]]; // Convert from [lon, lat] to [lat, lng]
          }
        }
      } else {
        // Bus is idle - position at a depot
        // Find the last trip this bus completed
        let lastCompletedTripIdx = -1;
        let lastDepotIdx = 0;
        
        for (const tripIdx of sortedTripIndices) {
          const tripIdxNum = parseInt(tripIdx, 10);
          const trip = trips[tripIdx];
          
          if (time >= trip.return) {
            lastCompletedTripIdx = tripIdxNum;
          } else {
            break;
          }
        }
        
        if (lastCompletedTripIdx >= 0 && busSchedule[lastCompletedTripIdx]) {
          // Position at the end depot of the last completed trip
          lastDepotIdx = busSchedule[lastCompletedTripIdx].end_depot;
        }
        
        const depotCoords = locationData[lastDepotIdx]?.coords || [0, 0];
        position = [depotCoords[1], depotCoords[0]]; // Convert from [lon, lat] to [lat, lng]
      }
      
      // Get current trip info for the popup
      let currentTripInfo = undefined;
      if (currentTripIdx !== null) {
        currentTripInfo = `Trip ${currentTripIdx}`;
      }
      
      positions.push({
        busId: busIdNumber,
        position,
        isMoving: busIsMoving,
        currentTrip: currentTripInfo,
        nextStop
      });
    });
    
    return positions;
  };

  // The actual animation function - separated completely to avoid closure issues
  function animationStep(timestamp: number) {
    // Initialize timestamps on first call
    if (prevTimestampRef.current === null) {
      prevTimestampRef.current = timestamp;
      rafId.current = requestAnimationFrame(animationStep);
      return;
    }
    
    const elapsed = timestamp - prevTimestampRef.current;
    prevTimestampRef.current = timestamp;
    
    // Calculate new time based on elapsed time and playback speed
    const timeStep = (elapsed / 1000) * playbackSpeed;
    let newTime = currentTime + timeStep;
    
    // Ensure we don't exceed the end time
    if (newTime >= endTime) {
      newTime = endTime;
      setIsPlaying(false);
    }
    
    // Update bus positions
    const positions = calculateBusPositions(newTime);
    onBusPositionsUpdate(positions);
    
    // Update facility statuses
    const statuses = calculateFacilityStatus(newTime);
    onFacilityStatusUpdate(statuses);
    
    // Update current time
    setCurrentTime(newTime);
    
    // Continue animation if still playing
    if (newTime < endTime) {
      rafId.current = requestAnimationFrame(animationStep);
    }
  }

  // Start or stop animation based on isPlaying state
  useEffect(() => {
    console.log("Animation state changed:", isPlaying);
    
    if (isPlaying) {
      // Start animation only if not at the end
      if (currentTime < endTime) {
        // Reset timestamps to ensure smooth start
        prevTimestampRef.current = null;
        rafId.current = requestAnimationFrame(animationStep);
        console.log("Animation started", rafId.current);
      } else {
        // If we're at the end, reset to start
        setCurrentTime(startTime);
        prevTimestampRef.current = null;
        rafId.current = requestAnimationFrame(animationStep);
        console.log("Animation restarted from beginning");
      }
    } else {
      // Stop animation
      if (rafId.current) {
        console.log("Animation stopped", rafId.current);
        cancelAnimationFrame(rafId.current);
        rafId.current = null;
      }
    }
    
    // Cleanup on component unmount
    return () => {
      if (rafId.current) {
        cancelAnimationFrame(rafId.current);
        rafId.current = null;
      }
    };
  }, [isPlaying, currentTime, endTime, playbackSpeed]);

  // When time is manually changed, update bus positions and facility statuses
  useEffect(() => {
    // Don't update positions during animation - the animation function handles that
    if (!isPlaying) {
      const positions = calculateBusPositions(currentTime);
      onBusPositionsUpdate(positions);
      
      const statuses = calculateFacilityStatus(currentTime);
      onFacilityStatusUpdate(statuses);
    }
  }, [currentTime, isPlaying]);

  // Handle timeline controls
  const handleTimeChange = (time: number) => {
    setCurrentTime(time);
  };

  const handlePlayPause = (playing: boolean) => {
    console.log("AnimationManager: Play/Pause called with:", playing);
    setIsPlaying(playing);
  };

  const handleSpeedChange = (speed: number) => {
    setPlaybackSpeed(speed);
  };

  // Calculate bus statuses for display
  const busStatuses = Object.keys(simulationData).map(busIdx => {
    const busIdNumber = parseInt(busIdx, 10);
    const positions = calculateBusPositions(currentTime);
    const busPosition = positions.find(p => p.busId === busIdNumber);
    
    return {
      busId: busIdNumber,
      isMoving: busPosition?.isMoving || false,
      currentTrip: busPosition?.currentTrip,
      nextStop: busPosition?.nextStop
    };
  });

  return (
    <div className="animation-panel">
      <h3>Animation Controls</h3>
      
      <TimelineController
        startTime={startTime}
        endTime={endTime}
        currentTime={currentTime}
        isPlaying={isPlaying}
        playbackSpeed={playbackSpeed}
        onTimeChange={handleTimeChange}
        onPlayPause={handlePlayPause}
        onSpeedChange={handleSpeedChange}
      />
      
      <div className="bus-status">
        {busStatuses.map(status => (
          <div 
            key={status.busId} 
            className={`bus-status-item ${status.isMoving ? 'in-transit' : 'idle'}`}
          >
            <div><strong>Vehicle {status.busId}</strong> {status.isMoving ? '🚌' : '🅿️'}</div>
            {status.currentTrip && (
              <div>{status.currentTrip}</div>
            )}
            {status.nextStop && (
              <div>Next: {status.nextStop}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default AnimationManager;
