// Path: frontend2/src/app/components/RouteControlPanel.tsx
'use client';

import React from 'react';
import { Trip } from './types';

interface RouteControlPanelProps {
  simulationData: any;
  bestSolution: Trip[][];
  selectedRoutes: { [key: string]: boolean };
  onRouteToggle: (routeKey: string, isSelected: boolean) => void;
  onBusToggle: (busIdx: string, isSelected: boolean) => void;
  onToggleAll: (isSelected: boolean) => void;
}

const RouteControlPanel: React.FC<RouteControlPanelProps> = ({
  simulationData,
  bestSolution,
  selectedRoutes,
  onRouteToggle,
  onBusToggle,
  onToggleAll
}) => {
  const [expandedBuses, setExpandedBuses] = React.useState<{ [key: string]: boolean }>({});

  // Check if all routes are selected
  const allRouteKeys = Object.keys(simulationData).flatMap(busIdx => 
    Object.keys(simulationData[busIdx]).map(tripIdx => `${busIdx}-${tripIdx}`)
  );
  const allRoutesSelected = allRouteKeys.length > 0 && allRouteKeys.every(key => selectedRoutes[key]);

  // Toggle bus expansion
  const toggleBusExpansion = (busIdx: string) => {
    setExpandedBuses(prev => ({
      ...prev,
      [busIdx]: !prev[busIdx]
    }));
  };

  // Check if all routes for a specific bus are selected
  const areBusRoutesSelected = (busIdx: string) => {
    const busRouteKeys = Object.keys(simulationData[busIdx] || {}).map(
      tripIdx => `${busIdx}-${tripIdx}`
    );
    
    return busRouteKeys.length > 0 && busRouteKeys.every(key => selectedRoutes[key]);
  };

  // Sort bus indices numerically
  const sortedBusIndices = Object.keys(simulationData).sort((a, b) => 
    parseInt(a) - parseInt(b)
  );

  return (
    <div className="route-control-panel">
      <div className="global-controls">
        <label className="toggle-all">
          <input
            type="checkbox"
            checked={allRoutesSelected}
            onChange={() => onToggleAll(!allRoutesSelected)}
          />
          <span>{allRoutesSelected ? 'Deselect All' : 'Select All'}</span>
        </label>
      </div>
      
      <div className="bus-list">
        {sortedBusIndices.map(busIdx => {
          const busData = simulationData[busIdx];
          const tripCount = Object.keys(busData || {}).length;
          const busSelected = areBusRoutesSelected(busIdx);
          const isBusExpanded = expandedBuses[busIdx] || false;
          
          if (tripCount === 0) return null;
          
          return (
            <div key={`bus-${busIdx}`} className="bus-item">
              <div className="bus-header">
                <label>
                  <input
                    type="checkbox"
                    checked={busSelected}
                    onChange={() => onBusToggle(busIdx, !busSelected)}
                  />
                  <span className="bus-title">Vehicle {busIdx}</span>
                  <span className="trip-count">({tripCount} trips)</span>
                </label>
                
                <button 
                  className="expand-button"
                  onClick={() => toggleBusExpansion(busIdx)}
                >
                  {isBusExpanded ? '−' : '+'}
                </button>
              </div>
              
              {isBusExpanded && (
                <div className="trip-list">
                  {Object.keys(busData || {})
                    .sort((a, b) => parseInt(a) - parseInt(b))
                    .map(tripIdx => {
                      const routeKey = `${busIdx}-${tripIdx}`;
                      const isSelected = selectedRoutes[routeKey] || false;
                      const tripObject = bestSolution[parseInt(busIdx)]?.[parseInt(tripIdx)];
                      const hasStops = tripObject && tripObject.stops.length > 0;
                      
                      return (
                        <div key={routeKey} className="trip-item">
                          <label>
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={() => onRouteToggle(routeKey, !isSelected)}
                            />
                            <span className="trip-title">Trip {tripIdx}</span>
                            {hasStops && (
                              <span className="stop-count">
                                ({tripObject.stops.length} stops)
                              </span>
                            )}
                          </label>
                        </div>
                      );
                    })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default RouteControlPanel;
