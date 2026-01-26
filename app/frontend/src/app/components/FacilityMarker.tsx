'use client';

import React from 'react';
import L from 'leaflet';
import { Marker, Popup } from 'react-leaflet';


interface FacilityMarkerProps {
  position: [number, number]; // [lat, lng]
  label: string;
  initialEvacuees: number;
  remainingEvacuees: number;
  facilityId: number;
}

// Create a custom div icon for the facilities that shows the remaining evacuees
const createFacilityIcon = (remainingEvacuees: number, initialEvacuees: number) => {
  // Determine if this facility has evacuees and what percentage remain
  const hasEvacuees = initialEvacuees > 0;
  const percentRemaining = hasEvacuees ? (remainingEvacuees / initialEvacuees) * 100 : 0;
  
  // Determine color based on percentage remaining
  let color = '#2ecc71'; // All evacuated (green)
  if (percentRemaining > 75) {
    color = '#e74c3c'; // Critical (red)
  } else if (percentRemaining > 50) {
    color = '#f39c12'; // Warning (orange)
  } else if (percentRemaining > 0) {
    color = '#3498db'; // In progress (blue)
  }
  
  // Determine size based on initial evacuee count (larger facilities = larger markers)
  const baseSize = 24;
  const sizeMultiplier = Math.min(1.0 + (initialEvacuees / 100) * 0.5, 2.0);
  const size = Math.round(baseSize * sizeMultiplier);
  
  return L.divIcon({
    className: '',
    iconSize: [size, size] as [number, number],
    iconAnchor: [size/2, size/2] as [number, number],
    html: `
      <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        width: ${size}px;
        height: ${size}px;
        border-radius: 50%;
        background-color: ${color};
        border: 2px solid white;
        color: white;
        font-weight: bold;
        font-size: ${size > 30 ? 12 : 10}px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
      ">
        ${remainingEvacuees}
      </div>
    `
  });
};

const FacilityMarker: React.FC<FacilityMarkerProps> = ({
  position,
  label,
  initialEvacuees,
  remainingEvacuees,
  facilityId
}) => {
  const icon = createFacilityIcon(remainingEvacuees, initialEvacuees);
  
  return (
    <Marker position={position} icon={icon}>
      <Popup>
        <div>
          <strong>{label}</strong>
          <div>ID: {facilityId}</div>
          <div>Initial Evacuees: {initialEvacuees}</div>
          <div>
            Remaining: {remainingEvacuees} 
            {initialEvacuees > 0 && ` (${Math.round((remainingEvacuees / initialEvacuees) * 100)}%)`}
          </div>
          {remainingEvacuees === 0 && initialEvacuees > 0 && (
            <div style={{ color: 'green', fontWeight: 'bold' }}>Fully Evacuated ✓</div>
          )}
        </div>
      </Popup>
    </Marker>
  );
};

export default FacilityMarker;