'use client';

import React from 'react';

interface MapLegendProps {
  visible: boolean;
}

const MapLegend: React.FC<MapLegendProps> = ({ visible }) => {
  if (!visible) return null;
  
  return (
    <div style={{
      position: 'absolute',
      bottom: '20px',
      right: '20px',
      backgroundColor: 'white',
      padding: '10px',
      borderRadius: '5px',
      boxShadow: '0 0 10px rgba(0,0,0,0.2)',
      zIndex: 1000,
      fontSize: '12px',
      maxWidth: '220px'
    }}>
      <h4 style={{ margin: '0 0 8px 0' }}>Facility Status</h4>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '5px' }}>
        <div style={{ 
          width: '16px', 
          height: '16px', 
          borderRadius: '50%', 
          backgroundColor: '#e74c3c',
          marginRight: '8px',
          border: '1px solid white',
          boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
        }}></div>
        <span>75-100% remaining (Critical)</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '5px' }}>
        <div style={{ 
          width: '16px', 
          height: '16px', 
          borderRadius: '50%', 
          backgroundColor: '#f39c12',
          marginRight: '8px',
          border: '1px solid white',
          boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
        }}></div>
        <span>50-75% remaining (Warning)</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '5px' }}>
        <div style={{ 
          width: '16px', 
          height: '16px', 
          borderRadius: '50%', 
          backgroundColor: '#3498db',
          marginRight: '8px',
          border: '1px solid white',
          boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
        }}></div>
        <span>1-50% remaining (In Progress)</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center' }}>
        <div style={{ 
          width: '16px', 
          height: '16px', 
          borderRadius: '50%', 
          backgroundColor: '#2ecc71',
          marginRight: '8px',
          border: '1px solid white',
          boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
        }}></div>
        <span>0% remaining (Evacuated)</span>
      </div>
      <div style={{ fontSize: '10px', marginTop: '8px', color: '#666' }}>
        The number displayed is the count of people in each facility.
        Marker size indicates facility capacity.
      </div>
    </div>
  );
};

export default MapLegend;