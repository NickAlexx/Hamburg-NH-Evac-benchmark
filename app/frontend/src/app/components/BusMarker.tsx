'use client';

import React, { useEffect, useState, useRef } from 'react';
import L from 'leaflet';
import { Marker, Popup, useMap } from 'react-leaflet';

interface BusMarkerProps {
  busId: number;
  position: [number, number]; // [lat, lng]
  isMoving: boolean;
  currentTrip?: string;
  nextStop?: string;
}

// Create a custom div icon for the bus
const createBusIcon = (busId: number, isMoving: boolean) => {
  return L.divIcon({
    className: '',
    iconSize: [30, 30] as [number, number],
    iconAnchor: [15, 15] as [number, number],
    html: `<div class="bus-marker ${isMoving ? 'in-transit' : 'idle'}">${busId}</div>`
  });
};

// Create a transform CSS rule for smooth movement
const createTransform = (angle: number) => {
  return `rotate(${angle}deg)`;
};

const BusMarker: React.FC<BusMarkerProps> = ({ 
  busId, 
  position, 
  isMoving, 
  currentTrip,
  nextStop 
}) => {
  // We need to track the last position for animation and heading calculation
  const [currentPosition, setCurrentPosition] = useState<[number, number]>(position);
  const lastPositionRef = useRef<[number, number]>(position);
  const lastUpdateTimeRef = useRef<number>(Date.now());

  // Track speed for icon styling (fast vs slow)
  const [movementSpeed, setMovementSpeed] = useState<number>(0);
  
  // Compute rotation angle based on movement direction
  const [rotationAngle, setRotationAngle] = useState<number>(0);
  
  // Use a custom icon for the bus
  const busIcon = createBusIcon(busId, isMoving);

  // When the position changes, update with smooth transition
  useEffect(() => {
    const now = Date.now();
    const timeDelta = now - lastUpdateTimeRef.current;
    lastUpdateTimeRef.current = now;
    
    // Calculate speed (distance/time)
    if (timeDelta > 0) {
      const lastLat = lastPositionRef.current[0];
      const lastLng = lastPositionRef.current[1];
      const newLat = position[0];
      const newLng = position[1];
      
      // Distance calculation (simplified)
      const dx = newLng - lastLng;
      const dy = newLat - lastLat;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      // Calculate speed - useful for styling
      const speed = distance / (timeDelta / 1000);
      setMovementSpeed(speed);
      
      // Calculate angle for rotation (if movement is significant)
      if (distance > 0.0001) {
        // Calculate angle in degrees (0 = North, 90 = East, etc.)
        let angle = Math.atan2(dx, dy) * (180 / Math.PI);
        setRotationAngle(angle);
      }
    }
    
    // If there's a significant change in position, update immediately
    if (Math.abs(position[0] - lastPositionRef.current[0]) > 0.01 ||
        Math.abs(position[1] - lastPositionRef.current[1]) > 0.01) {
      setCurrentPosition(position);
      lastPositionRef.current = position;
      return;
    }
    
    // Otherwise update smoothly
    setCurrentPosition(position);
    lastPositionRef.current = position;
  }, [position]);

  // Add CSS class based on movement state
  const getMarkerClasses = () => {
    if (!isMoving) return 'idle';
    if (movementSpeed > 0.0005) return 'fast';
    return 'moving';
  };

  // Enhanced popup content
  const renderPopupContent = () => (
    <div>
      <strong>Vehicle {busId}</strong>
      <div>Status: {isMoving ? 'In Transit' : 'Idle'}</div>
      {currentTrip && <div>Current Trip: {currentTrip}</div>}
      {nextStop && <div>Next Stop: {nextStop}</div>}
      {isMoving && <div>Speed: {(movementSpeed * 1000).toFixed(2)} units/s</div>}
    </div>
  );

  return (
    <Marker 
      position={currentPosition} 
      icon={busIcon}
    >
      <Popup>
        {renderPopupContent()}
      </Popup>
    </Marker>
  );
};

export default BusMarker;
