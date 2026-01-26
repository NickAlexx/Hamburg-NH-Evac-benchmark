// Path: frontend2/src/app/components/SimulationParameters.tsx
'use client';

import React, { useState, useMemo } from 'react';

// --- TYPE DEFINITIONS (centralized here for this component) ---

// Represents the state of a vehicle being configured in the UI
export interface Vehicle {
  id: string; // Unique ID for React keys
  capacity: number;
  start:
  | { kind: 'depot', index: number }
  | { kind: 'node', index: number }
  | { kind: 'coord', coords: [number, number] | null }; // [lon, lat] or null if not set
}

// Represents the payload structure for a single vehicle sent to the backend
export interface BackendVehiclePayload {
  capacity: number;
  start_depot?: number;
  start_node?: number;
  start_coord?: { lat: number; lon: number };
}

export interface ALNSConfig {
  use_memetic_polish?: boolean;
}

interface Location {
  label: string;
  coords: [number, number];
  people: number;
}

interface SimulationParametersProps {
  onParametersChange: (params: SimulationParams) => void;
  isRunning: boolean;
  onRunSimulation: () => void;
  onSetMainCenterMode: (isActive: boolean) => void;
  isSetMainCenterMode: boolean;
  customMainCenterCoords: [number, number] | null;
  onClearMainCenter: () => void;
  depots: Location[];
  facilities: Location[];
  placingVehicleId: string | null;
  onSetVehiclePlacementMode: (vehicleId: string | null) => void;
  // Props for lifted state
  useCustomFleet: boolean;
  onUseCustomFleetChange: (value: boolean) => void;
  vehicles: Vehicle[];
  onVehiclesChange: (vehicles: Vehicle[]) => void;
  onLoadBombPreset: () => void;
  onLoadFloodPreset: () => void;
  onLoadDefaultPreset: () => void; // New prop
  currentParams: SimulationParams;
}

export interface SimulationParams {
  algorithm_choice: 'ea2' | 'ea2_memetic' | 'pendelverkehr' | 'alns';
  buses_count: number;
  bus_capacity: number;
  time_limit_seconds: number;
  population_size?: number;
  buffer_meters: number;
  vehicles?: BackendVehiclePayload[];
  use_dynamic_service_time?: boolean;
  service_time_base_min?: number;
  service_time_per_person_seconds?: number;
  alns_config?: ALNSConfig;
}

const SimulationParameters: React.FC<SimulationParametersProps> = ({
  onParametersChange,
  isRunning,
  onRunSimulation,
  onSetMainCenterMode,
  isSetMainCenterMode,
  customMainCenterCoords,
  onClearMainCenter,
  depots,
  facilities,
  placingVehicleId,
  onSetVehiclePlacementMode,
  useCustomFleet,
  onUseCustomFleetChange,
  vehicles,
  onVehiclesChange,
  onLoadBombPreset,
  onLoadFloodPreset,
  onLoadDefaultPreset, // New prop
  currentParams,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;

    let processedValue: string | number | boolean = value;

    if (type === 'checkbox') {
      processedValue = (e.target as HTMLInputElement).checked;
    } else if (type === 'number') {
      processedValue = parseFloat(value);
      if (name === 'runtime_minutes') {
        // Special handling for runtime in minutes
        onParametersChange({
          ...currentParams,
          time_limit_seconds: Math.round(processedValue * 60)
        });
        return;
      }
    }

    onParametersChange({
      ...currentParams,
      [name]: processedValue,
    });
  };

  const isEA = currentParams.algorithm_choice === 'ea2' || currentParams.algorithm_choice === 'ea2_memetic';
  const isAdvancedSolver = isEA || currentParams.algorithm_choice === 'pendelverkehr' || currentParams.algorithm_choice === 'alns';
  const mainCenterButtonText = useMemo(() => isSetMainCenterMode ? 'Click on Map...' : 'Set on Map', [isSetMainCenterMode]);

  const addVehicle = () => {
    const newVehicle: Vehicle = { id: crypto.randomUUID(), capacity: 80, start: { kind: 'depot', index: 0 } };
    onVehiclesChange([...vehicles, newVehicle]);
  };

  const removeVehicle = (id: string) => {
    onVehiclesChange(vehicles.filter(v => v.id !== id));
  };

  const updateVehicle = (id: string, field: 'capacity' | 'start.kind' | 'start.index', value: string) => {
    onVehiclesChange(
      vehicles.map((v): Vehicle => {
        if (v.id !== id) return v;
        if (field === 'capacity') return { ...v, capacity: parseInt(value, 10) || 0 };
        if (field === 'start.kind') {
          const newKind = value as 'depot' | 'node' | 'coord';
          if (v.start.kind === newKind) return v;
          // *** FIX: Changed 'let' to 'const' here ***
          const newStart: Vehicle['start'] = newKind === 'coord' ? { kind: 'coord', coords: null } : { kind: newKind, index: 0 };
          return { ...v, start: newStart };
        }
        if (field === 'start.index' && (v.start.kind === 'depot' || v.start.kind === 'node')) {
          return { ...v, start: { ...v.start, index: parseInt(value, 10) || 0 } };
        }
        return v;
      })
    );
  };

  return (
    <div className="simulation-params-compact">
      <div className="params-header">
        <h3>Parameters</h3>
        <button className="toggle-advanced-button" onClick={() => setShowAdvanced(!showAdvanced)}>{showAdvanced ? "−" : "+"}</button>
      </div>

      <div className="params-container">
        <div className="params-row" style={{ borderBottom: '1px solid #dee2e6', paddingBottom: '10px', marginBottom: '10px' }}>
          <div className="action-buttons-inline" style={{ width: '100%' }}>
            <button onClick={onLoadBombPreset} disabled={isRunning} className="mini-button" style={{ flex: '1', backgroundColor: '#f39c12', color: 'white' }}>Bomb Evac</button>
            <button onClick={onLoadFloodPreset} disabled={isRunning} className="mini-button" style={{ flex: '1', backgroundColor: '#3498db', color: 'white' }}>Flood WB</button>
            <button onClick={onLoadDefaultPreset} disabled={isRunning} className="mini-button" style={{ flex: '1', backgroundColor: '#6c757d', color: 'white' }}>Default</button>
          </div>
        </div>

        <div className="params-row">
          <div className="param-item compact" style={{ flex: '1' }}>
            <label htmlFor="algorithm_choice">Algorithm:</label>
            <select id="algorithm_choice" name="algorithm_choice" value={currentParams.algorithm_choice} onChange={handleChange} disabled={isRunning} className="algo-select" style={{ width: '100%' }}>
              <option value="ea2_memetic">EA + Memetic</option>
              <option value="ea2">EA (Standard)</option>
              <option value="pendelverkehr">Shuttle (Baseline)</option>
              <option value="alns">ALNS (Baseline)</option>
            </select>
          </div>
        </div>

        {isAdvancedSolver && (
          <div className="fleet-toggle">
            <label>
              <input type="radio" name="fleetType" checked={!useCustomFleet} onChange={() => onUseCustomFleetChange(false)} disabled={isRunning} />
              <span>Simple Fleet</span>
            </label>
            <label>
              <input type="radio" name="fleetType" checked={useCustomFleet} onChange={() => onUseCustomFleetChange(true)} disabled={isRunning} />
              <span>Custom Fleet</span>
            </label>
          </div>
        )}

        {(!isAdvancedSolver || !useCustomFleet) && (
          <div className="params-row">
            <div className="param-item compact">
              <label htmlFor="buses_count">Vehicles:</label>
              <input type="number" id="buses_count" name="buses_count" value={currentParams.buses_count} onChange={handleChange} min="1" max="20" disabled={isRunning} className="mini-input" />
            </div>
            <div className="param-item compact">
              <label htmlFor="bus_capacity">Capacity:</label>
              <input type="number" id="bus_capacity" name="bus_capacity" value={currentParams.bus_capacity} onChange={handleChange} min="10" max="200" disabled={isRunning} className="mini-input" />
            </div>
          </div>
        )}

        {isAdvancedSolver && useCustomFleet && (
          <div className="custom-fleet-editor">
            <h4>Custom Vehicle Fleet ({vehicles.length})</h4>
            <div className="vehicle-grid">
              {vehicles.map((vehicle, index) => (
                <div key={vehicle.id} className="vehicle-row">
                  <span className="vehicle-label">V{index}</span>
                  <input type="number" value={vehicle.capacity} onChange={e => updateVehicle(vehicle.id, 'capacity', e.target.value)} placeholder="Cap" className="mini-input" disabled={isRunning} />
                  <select value={vehicle.start.kind} onChange={e => updateVehicle(vehicle.id, 'start.kind', e.target.value)} className="mini-select" disabled={isRunning}>
                    <option value="depot">Depot</option>
                    <option value="node">Facility</option>
                    <option value="coord">Coordinate</option>
                  </select>

                  {vehicle.start.kind === 'coord' ? (
                    <div className="coord-control">
                      <button onClick={() => onSetVehiclePlacementMode(placingVehicleId === vehicle.id ? null : vehicle.id)} className={`set-on-map-btn ${placingVehicleId === vehicle.id ? 'active' : ''}`} disabled={isRunning}>
                        {placingVehicleId === vehicle.id ? 'Placing...' : 'Set on Map'}
                      </button>
                      {vehicle.start.coords && <span>({vehicle.start.coords[1].toFixed(2)}, {vehicle.start.coords[0].toFixed(2)})</span>}
                    </div>
                  ) : (
                    <select value={vehicle.start.index} onChange={e => updateVehicle(vehicle.id, 'start.index', e.target.value)} className="mini-select" disabled={isRunning}>
                      {vehicle.start.kind === 'depot' ?
                        (depots.length > 0 ?
                          depots.map((d, i) => <option key={`d-${i}`} value={i}>{d.label.split(',')[0]}</option>) :
                          <option disabled>Add a depot on map</option>
                        ) :
                        facilities.map((f, i) => <option key={`f-${i}`} value={i}>{`F${i}: ${f.label.split(',')[0]}`}</option>)
                      }
                    </select>
                  )}

                  <button onClick={() => removeVehicle(vehicle.id)} className="remove-vehicle-btn" disabled={isRunning || vehicles.length <= 1}>×</button>
                </div>
              ))}
            </div>
            <button onClick={addVehicle} className="add-vehicle-btn" disabled={isRunning}>+ Add Vehicle</button>
          </div>
        )}

        <div className="params-row">
          <div className="param-item compact">
            <label htmlFor="buffer_meters">Buffer (m):</label>
            <input type="number" id="buffer_meters" name="buffer_meters" value={currentParams.buffer_meters} onChange={handleChange} min="100" disabled={isRunning} className="mini-input" />
          </div>
          <div className="param-item compact">
            <label htmlFor="runtime_minutes">Runtime (min):</label>
            <input type="number" id="runtime_minutes" name="runtime_minutes" value={currentParams.time_limit_seconds / 60} onChange={handleChange} min="0.1" max="10" step="0.1" disabled={isRunning} className="mini-input" />
          </div>
        </div>

        {showAdvanced && (
          <div className="advanced-options">
            {isEA && (
              <div className="params-row advanced">
                <div className="param-item compact">
                  <label htmlFor="population_size">Population:</label>
                  <input type="number" id="population_size" name="population_size" value={currentParams.population_size || 200} onChange={handleChange} min="50" max="5000" step="50" disabled={isRunning || !isEA} />
                </div>
              </div>
            )}
            <div className="params-row advanced service-time-model">
              <label className="checkbox-label">
                <input type="checkbox" name="use_dynamic_service_time" checked={!!currentParams.use_dynamic_service_time} onChange={handleChange} disabled={isRunning} />
                Use Dynamic Service Time
              </label>
              {currentParams.use_dynamic_service_time && (
                <div className="dynamic-service-inputs">
                  <div className="param-item compact">
                    <label htmlFor="service_time_base_min">Base (min):</label>
                    <input type="number" id="service_time_base_min" name="service_time_base_min" value={currentParams.service_time_base_min || 3.0} onChange={handleChange} min="0" step="0.5" disabled={isRunning} className="mini-input" />
                  </div>
                  <div className="param-item compact">
                    <label htmlFor="service_time_per_person_seconds">Per Person (sec):</label>
                    <input type="number" id="service_time_per_person_seconds" name="service_time_per_person_seconds" value={currentParams.service_time_per_person_seconds || 20} onChange={handleChange} min="0" disabled={isRunning} className="mini-input" />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <div className="params-row main-center-controls">
          <div className="param-item compact">
            <label>Buffer Center:</label>
            <div className="main-center-info">
              {customMainCenterCoords ? (
                <>
                  <span>{`(${customMainCenterCoords[1].toFixed(3)}, ${customMainCenterCoords[0].toFixed(3)})`}</span>
                  <button onClick={onClearMainCenter} className="clear-center-btn" title="Reset to default">×</button>
                </>
              ) : (
                <span>Default</span>
              )}
            </div>
          </div>
          <button onClick={() => onSetMainCenterMode(!isSetMainCenterMode)} className={`set-on-map-btn ${isSetMainCenterMode ? 'active' : ''}`} disabled={isRunning}>{mainCenterButtonText}</button>
        </div>

        <div className="params-row">
          <div className="action-buttons-inline" style={{ width: '100%' }}>
            <button onClick={onRunSimulation} disabled={isRunning} className="mini-button run-button" style={{ flex: '1' }}>
              {isRunning ? 'Running...' : 'Run Simulation'}
            </button>
          </div>
        </div>
      </div>
      <style jsx>{`
        .advanced-options {
          display: flex;
          flex-direction: column;
          gap: 8px;
          margin-top: 8px;
          padding-top: 8px;
          border-top: 1px solid #dee2e6;
        }
        .service-time-model {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          background-color: #f0f0f0;
          padding: 8px;
          border-radius: 6px;
        }
        .checkbox-label {
          display: flex;
          align-items: center;
          gap: 8px;
          font-weight: 500;
          font-size: 0.85rem;
        }
        .dynamic-service-inputs {
          display: flex;
          gap: 15px;
          margin-top: 8px;
          padding-left: 20px;
        }
      `}</style>
    </div>
  );
};

export default SimulationParameters;
