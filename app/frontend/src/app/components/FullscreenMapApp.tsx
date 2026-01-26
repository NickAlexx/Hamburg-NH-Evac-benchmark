// Path: frontend2/src/app/components/FullscreenMapApp.tsx
// ========================================
// THIS FILE HAS BEEN MODIFIED FOR DATA CONSISTENCY
// ========================================
'use client';

import React, { useState, useMemo, useEffect, useRef } from 'react';
import axios from 'axios';
import dynamic from 'next/dynamic';

// Import context
import { useEvacuation } from '../context/EvacuationContext';

// Import custom hook for draggable panels
import { useDraggablePanel } from '../hooks/useDraggablePanel';

// Import components
import SimulationParameters, { SimulationParams } from './SimulationParameters';
import BusGanttChart from './BusGanttChart';
import RouteControlPanel from './RouteControlPanel';
import AllRoutesView from './AllRoutesView';

// Import scenario definitions
import { BOMB_EVACUATION_PRESET, WILHELMSBURG_FLOOD_PRESET, DEFAULT_SCENARIO_PRESET } from '../data/scenarios';

// Dynamically import components that require the window object
const LeafletMap = dynamic(() => import('./LeafletMap'), {
  ssr: false,
  loading: () => <div className="loading-container">Loading Map...</div>
});

// Type imports (removed unused BusPosition and FacilityStatus)
import { Trip, EvacuationMetrics, UserEvacCenter, BackendVehicle } from './types';
// Import new vehicle types directly since they are needed for state
import { Vehicle, BackendVehiclePayload } from './SimulationParameters';

// NEW CONSTANT: Read the API base URL from environment variables
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;

interface Location {
  label: string;
  coords: [number, number]; // (lon, lat)
  people: number;
  capacity?: number | null;
}

interface TripSimulation {
  route?: [number, number][];
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

interface MetricsData {
  wait?: {
    mean_min?: number;
    sum_person_minutes?: number;
  };
  timeline?: {
    latest_return_min?: number;
  };
  counts?: {
    people_picked?: number;
  };
}

interface EAResult {
  overall_cost: number;
  best_solution: Trip[][];
  simulation_data: SimulationData;
  facility_timeline_features: Record<string, unknown>[];
  depots: Location[];
  facilities: Location[];
  num_buses: number;
  algorithm: string;
  timestamp?: string;
  logs_directory?: string;
  vehicles?: BackendVehicle[]; // API response vehicle format
  metrics?: MetricsData;
}

interface OptimizationJobStatus {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  result?: EAResult | null;
  error?: string | null;
}

// Default values
const defaultLocationData: Location[] = [
  {
    label: "Default Evacuation Center",
    coords: [9.996754980861652, 53.49221335731889],
    people: 0
  }
];
const defaultSimulationData: SimulationData = {};
const defaultBestSolution: Trip[][] = [];
const defaultSelectedRoutes: { [key: string]: boolean } = {};


// Main App Component
const FullscreenMapApp: React.FC = () => {
  // Get context functions
  const {
    setSimulationData: setContextSimulationData,
    setBestSolution: setContextBestSolution,
    depots: contextDepots,
    setDepots: setContextDepots,
    facilities: contextFacilities,
    setFacilities: setContextFacilities,
    setRoutePaths: setContextRoutePaths,
    userEvacCenters,
    addUserEvacCenter,
    removeUserEvacCenter,
    updateUserEvacCenter, // *** NEW: Get the updater function ***
    setUserEvacCenters // *** NEW: Get the setter for all user centers ***
  } = useEvacuation();

  // State
  const [data, setData] = useState<EAResult | null>(null);
  const [selectedRoutes, setSelectedRoutes] = useState<{ [key: string]: boolean }>(defaultSelectedRoutes);
  const [loading, setLoading] = useState<boolean>(false);

  // *** NEW STATE: To manage on-demand route fetching ***
  const [routesToFetch, setRoutesToFetch] = useState<string[]>([]);

  // UI state for collapsible panels
  const [showParameters, setShowParameters] = useState<boolean>(true);
  const [showRouteControls, setShowRouteControls] = useState<boolean>(true); // Show by default
  const [showMetrics, setShowMetrics] = useState<boolean>(true);
  const [resultsScale, setResultsScale] = useState<number>(1);

  // Store simulation parameters
  const [simulationParams, setSimulationParams] = useState<SimulationParams>({
    algorithm_choice: 'alns',
    buses_count: 3,
    bus_capacity: 80,
    time_limit_seconds: 30, // Default to 30 seconds
    population_size: 200,
    buffer_meters: 5000,
    use_dynamic_service_time: true,
    service_time_base_min: 3.0,
    service_time_per_person_seconds: 20,
    alns_config: { use_memetic_polish: false },
  });

  // --- LIFTED STATE from SimulationParameters ---
  const [useCustomFleet, setUseCustomFleet] = useState(false);
  const [vehicles, setVehicles] = useState<Vehicle[]>([
    { id: crypto.randomUUID(), capacity: 80, start: { kind: 'depot', index: 0 } },
    { id: crypto.randomUUID(), capacity: 80, start: { kind: 'depot', index: 0 } },
    { id: crypto.randomUUID(), capacity: 80, start: { kind: 'depot', index: 0 } },
  ]);
  const [placingVehicleId, setPlacingVehicleId] = useState<string | null>(null);
  // --- END LIFTED STATE ---

  // Evacuation metrics
  const [evacuationMetrics, setEvacuationMetrics] = useState<EvacuationMetrics>({
    averageEvacTime: 0,
    latestEvacTime: 0,
    cumulativeWaitingTime: 0,
    totalEvacuees: 0
  });

  // State for facilities that load on startup
  const [preloadedFacilities, setPreloadedFacilities] = useState<Location[]>([]);

  // State for the AllRoutesView
  const [showAllRoutes, setShowAllRoutes] = useState<boolean>(false);

  // State for GanttChart visibility
  const [showGanttChart, setShowGanttChart] = useState<boolean>(false);

  // Refs for draggable panels
  const parametersRef = useRef<HTMLDivElement>(null);
  const routeControlsRef = useRef<HTMLDivElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const ganttRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!showMetrics || !resultsRef.current) return;
    const panel = resultsRef.current;
    const baseWidth = 320;
    const baseHeight = 260;
    const minScale = 0.7;
    const maxScale = 1;

    const updateScale = () => {
      const widthScale = panel.clientWidth / baseWidth;
      const heightScale = panel.clientHeight / baseHeight;
      const nextScale = Math.min(widthScale, heightScale, maxScale);
      const clampedScale = Math.max(minScale, nextScale);
      setResultsScale(parseFloat(clampedScale.toFixed(2)));
    };

    updateScale();
    const observer = new ResizeObserver(updateScale);
    observer.observe(panel);

    return () => observer.disconnect();
  }, [showMetrics, data]);

  // NEW state for custom center/buffer
  const [isSetMainCenterMode, setIsSetMainCenterMode] = useState(false);
  const [customMainCenter, setCustomMainCenter] = useState<[number, number] | null>(null);

  // Apply draggable panel behavior
  useDraggablePanel(parametersRef, {
    handleSelector: '.panel-header',
    bounds: { top: 10, right: 10, bottom: 10, left: 10 }
  });

  useDraggablePanel(routeControlsRef, {
    handleSelector: '.panel-header',
    bounds: { top: 10, right: 10, bottom: 10, left: 10 }
  });

  useDraggablePanel(resultsRef, {
    handleSelector: '.panel-header',
    bounds: { top: 10, right: 10, bottom: 10, left: 10 }
  });

  useDraggablePanel(ganttRef, {
    handleSelector: '.panel-header',
    bounds: { top: 10, right: 10, bottom: 10, left: 10 }
  });

  // Unified hook for fetching facilities
  useEffect(() => {
    const { buffer_meters } = simulationParams;

    const fetchFacilities = async () => {
      try {
        const query = new URLSearchParams();
        if (customMainCenter) {
          query.append('center_lon', customMainCenter[0].toString());
          query.append('center_lat', customMainCenter[1].toString());
        }
        if (buffer_meters) {
          query.append('buffer_m', buffer_meters.toString());
        }

        const response = await axios.get(`${API_BASE_URL}/facilities?${query.toString()}`);
        const { facilities, depots } = response.data;

        setPreloadedFacilities(facilities);
        setContextFacilities(facilities);

        if (depots && depots.length > 0) {
          setContextDepots(depots);
        }
      } catch (error) {
        console.error('Error fetching facilities:', error);
        handleAPIError(error);
      }
    };

    const timer = setTimeout(() => {
      fetchFacilities();
    }, 300);

    return () => clearTimeout(timer);
  }, [customMainCenter, simulationParams, setContextDepots, setContextFacilities]);

  // *** NEW: Function to load the preset scenario ***
  const loadBombEvacuationPreset = () => {
    // 1. Set the main center (bomb location) and buffer
    setCustomMainCenter(BOMB_EVACUATION_PRESET.mainCenter);
    setSimulationParams(prev => ({
      ...prev,
      buffer_meters: BOMB_EVACUATION_PRESET.bufferMeters,
      // You might want to set a specific algorithm for this preset
      algorithm_choice: 'alns',
      alns_config: { use_memetic_polish: false },
    }));

    // 2. Clear existing user evacuation centers and add the new one
    setUserEvacCenters([BOMB_EVACUATION_PRESET.evacCenter]);

    // 3. Configure the custom fleet
    setUseCustomFleet(true);
    const newVehicles: Vehicle[] = [];
    BOMB_EVACUATION_PRESET.fleet.forEach((group: { capacity: number; count: number }) => {
      for (let i = 0; i < group.count; i++) {
        newVehicles.push({
          id: crypto.randomUUID(),
          capacity: group.capacity,
          start: {
            kind: 'coord',
            coords: BOMB_EVACUATION_PRESET.vehicleStart,
          },
        });
      }
    });
    setVehicles(newVehicles);

    // 4. Provide user feedback
    alert('Szenario "Bombenevakuierung" geladen!');
  };

  // *** NEW: Function to load the Wilhelmsburg flood scenario ***
  const loadWilhelmsburgFloodPreset = () => {
    // 1. Set the main center and buffer
    setCustomMainCenter(WILHELMSBURG_FLOOD_PRESET.mainCenter);
    setSimulationParams(prev => ({
      ...prev,
      buffer_meters: WILHELMSBURG_FLOOD_PRESET.bufferMeters,
      algorithm_choice: 'alns',
      alns_config: { use_memetic_polish: false },
    }));

    // 2. Set the evacuation centers (multiple this time)
    setUserEvacCenters(WILHELMSBURG_FLOOD_PRESET.evacCenters);

    // 3. Configure the custom fleet
    setUseCustomFleet(true);
    const newVehicles: Vehicle[] = [];
    WILHELMSBURG_FLOOD_PRESET.fleet.forEach((group: { capacity: number; count: number }) => {
      for (let i = 0; i < group.count; i++) {
        newVehicles.push({
          id: crypto.randomUUID(),
          capacity: group.capacity,
          start: {
            kind: 'coord',
            coords: WILHELMSBURG_FLOOD_PRESET.vehicleStart,
          },
        });
      }
    });
    setVehicles(newVehicles);

    // 4. Provide user feedback
    alert('Szenario "Überschwemmung Wilhelmsburg" geladen!');
  };

  // *** NEW: Function to load the Default scenario ***
  const loadDefaultScenarioPreset = (showAlert: boolean = true) => {
    setCustomMainCenter(DEFAULT_SCENARIO_PRESET.mainCenter);
    setSimulationParams(prev => ({
      ...prev,
      buffer_meters: DEFAULT_SCENARIO_PRESET.bufferMeters,
      algorithm_choice: 'alns',
      alns_config: { use_memetic_polish: false },
    }));
    setUserEvacCenters(DEFAULT_SCENARIO_PRESET.evacCenters);
    setUseCustomFleet(true);

    const newVehicles: Vehicle[] = [];
    DEFAULT_SCENARIO_PRESET.fleet.forEach(group => {
      for (let i = 0; i < group.count; i++) {
        newVehicles.push({
          id: crypto.randomUUID(),
          capacity: group.capacity,
          start: DEFAULT_SCENARIO_PRESET.vehicleStart
            ? { kind: 'coord', coords: DEFAULT_SCENARIO_PRESET.vehicleStart }
            : { kind: 'depot', index: 0 }
        });
      }
    });
    setVehicles(newVehicles);
    if (showAlert) {
      alert('Default Scenario geladen!');
    }
  };

  useEffect(() => {
    loadDefaultScenarioPreset(false);
  }, []);

  // Merge data.depots + data.facilities for display
  const mergedLocationData = useMemo<Location[]>(() => {
    // FIX 1: Prioritize user-defined evac centers over old simulation results
    const effectiveDepots = userEvacCenters.length > 0
      ? userEvacCenters.map(zone => ({ label: zone.label, coords: zone.coords, people: 0, capacity: zone.capacity }))
      : (contextDepots.length > 0 ? [...contextDepots] : defaultLocationData);

    const facilities = data ? data.facilities : preloadedFacilities;

    return [
      ...effectiveDepots,
      ...facilities
    ];
  }, [data, userEvacCenters, preloadedFacilities, contextDepots]);

  // Total number of depots
  const totalDepotCount = userEvacCenters.length > 0
    ? userEvacCenters.length
    : (contextDepots.length > 0 ? contextDepots.length : defaultLocationData.length);

  const defaultDepotsCount = contextDepots.length > 0 ? contextDepots.length : defaultLocationData.length;

  const handleAddEvacCenter = (center: UserEvacCenter) => {
    const newLabel = `User Depot ${userEvacCenters.length + 1}`;
    addUserEvacCenter({ ...center, label: newLabel });
  };

  const handleRemoveEvacCenter = (index: number) => {
    removeUserEvacCenter(index);
  };

  // *** NEW: Handler for updating a center's properties (like capacity) ***
  const handleUpdateEvacCenter = (index: number, centerUpdate: Partial<UserEvacCenter>) => {
    updateUserEvacCenter(index, centerUpdate);
  };

  // New handler for setting the main center
  const handleSetMainCenter = (coords: [number, number]) => {
    setCustomMainCenter(coords);
    setIsSetMainCenterMode(false); // Turn off mode after selection
  };

  // NEW: Handler for setting vehicle coordinate on map click
  const handleSetVehicleCoordinate = (vehicleId: string, coords: [number, number]) => {
    setVehicles(prev =>
      prev.map(v =>
        v.id === vehicleId && v.start.kind === 'coord'
          ? { ...v, start: { kind: 'coord', coords: coords } }
          : v
      )
    );
    setPlacingVehicleId(null); // Exit placement mode
  };

  const handleClearMainCenter = () => {
    setCustomMainCenter(null);
  };

  const handleAPIError = (error: unknown) => {
    console.error('API Error:', error);

    let errorMessage = 'An unknown error occurred';

    if (axios.isAxiosError(error)) {
      if (error.response) {
        console.error('Error response:', error.response.data);

        if (error.response.data && error.response.data.detail) {
          if (typeof error.response.data.detail === 'object') {
            errorMessage = `Error: ${error.response.data.detail.error || 'Unknown server error'}`;
          } else {
            errorMessage = `Error: ${error.response.data.detail}`;
          }
        } else {
          errorMessage = `Server error (${error.response.status}): ${error.message}`;
        }
      } else if (error.request) {
        errorMessage = 'No response from server. Please check your connection.';
      } else {
        errorMessage = error.message || 'Error setting up the request';
      }
    } else if (error instanceof Error) {
      errorMessage = error.message;
    }

    alert(errorMessage);
    return errorMessage;
  };

  const handleParametersChange = (params: SimulationParams) => {
    setSimulationParams(params);
  };

  const runEvacuation = async () => {
    setLoading(true);

    try {
      console.log('Clearing old routes before new simulation...');
      setContextRoutePaths({});

      const isEA = simulationParams.algorithm_choice === 'ea2' || simulationParams.algorithm_choice === 'ea2_memetic';

      // *** MODIFIED ***: Define which algorithms can use custom fleet
      const canUseCustomFleet = isEA || simulationParams.algorithm_choice === 'pendelverkehr' || simulationParams.algorithm_choice === 'alns';

      const requestParams: Record<string, unknown> = {
        ...simulationParams,
        evacuation_zones: userEvacCenters,
        default_evac_center_coords: customMainCenter,
      };

      if (simulationParams.algorithm_choice === 'pendelverkehr') {
        delete requestParams.population_size;
      }

      // *** MODIFIED ***: Use the new check here
      if (useCustomFleet && canUseCustomFleet) {
        const backendVehicles: BackendVehiclePayload[] = vehicles.map(v => {
          const payload: BackendVehiclePayload = { capacity: v.capacity };
          if (v.start.kind === 'depot') {
            payload.start_depot = v.start.index;
          } else if (v.start.kind === 'node') {
            payload.start_node = v.start.index;
          } else if (v.start.kind === 'coord' && v.start.coords) {
            payload.start_coord = { lon: v.start.coords[0], lat: v.start.coords[1] };
          }
          return payload;
        });
        requestParams.vehicles = backendVehicles;
        requestParams.buses_count = vehicles.length; // Ensure bus count matches fleet size
      } else {
        delete requestParams.vehicles;
      }

      // The simulationParams state already includes the new service time fields,
      // so they are automatically sent to the backend here.
      const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
      const statusPollIntervalMs = 6000;

      const startResponse = await axios.post(`${API_BASE_URL}/optimization-jobs`, requestParams);
      const jobId: string | undefined = startResponse.data?.job_id;
      if (!jobId) {
        throw new Error('Backend did not return a job_id');
      }

      const maxWaitMs = (simulationParams.time_limit_seconds + 120) * 1000;
      const startedPollingAt = Date.now();

      let result: EAResult | null = null;
      while (true) {
        if (Date.now() - startedPollingAt > maxWaitMs) {
          throw new Error('Optimization timed out while waiting for result');
        }

        const statusResponse = await axios.get(`${API_BASE_URL}/optimization-jobs/${jobId}`);
        const job = statusResponse.data as OptimizationJobStatus;

        if (job.status === 'completed') {
          if (!job.result) {
            throw new Error('Optimization completed but no result was returned');
          }
          result = job.result;
          break;
        }

        if (job.status === 'failed') {
          throw new Error(job.error || 'Optimization failed');
        }

        await sleep(statusPollIntervalMs);
      }

      if (!result) {
        throw new Error('Optimization did not return a result');
      }
      setData(result);

      setContextSimulationData(result.simulation_data);
      setContextBestSolution(result.best_solution);
      setContextDepots(result.depots);
      setContextFacilities(result.facilities);

      // *** MODIFIED LOGIC: Use the precise metrics from the backend response ***
      if (result.metrics) {
        setEvacuationMetrics({
          averageEvacTime: result.metrics.wait?.mean_min || 0,
          latestEvacTime: result.metrics.timeline?.latest_return_min || 0,
          cumulativeWaitingTime: result.metrics.wait?.sum_person_minutes || 0,
          totalEvacuees: result.metrics.counts?.people_picked || 0
        });
      } else {
        // Fallback to resetting metrics if the key is missing
        console.warn("API response was missing the 'metrics' object. Metrics will be zero.");
        setEvacuationMetrics({
          averageEvacTime: 0,
          latestEvacTime: 0,
          cumulativeWaitingTime: 0,
          totalEvacuees: 0
        });
      }

      setSelectedRoutes({});
      setRoutesToFetch([]);

      setShowMetrics(true);
      setShowRouteControls(true); // Ensure route controls are visible after a run

    } catch (error) {
      handleAPIError(error);
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const handleRouteToggle = (key: string, isSelected: boolean) => {
    setSelectedRoutes(prev => ({ ...prev, [key]: isSelected }));

    if (isSelected) {
      setRoutesToFetch(prev => Array.from(new Set([...prev, key])));
    }
  };

  const handleBusToggle = (busIdx: string, isSelected: boolean) => {
    if (!data) return;
    const tripsForBus = Object.keys(data.simulation_data[busIdx]).map(tripIdx => `${busIdx}-${tripIdx}`);

    const newSelections = { ...selectedRoutes };
    tripsForBus.forEach(key => { newSelections[key] = isSelected; });
    setSelectedRoutes(newSelections);

    if (isSelected) {
      setRoutesToFetch(prev => Array.from(new Set([...prev, ...tripsForBus])));
    }
  };

  const handleToggleAllRoutes = (isSelected: boolean) => {
    if (!data) return;
    const allTripKeys: string[] = [];
    Object.keys(data.simulation_data).forEach(busIdx => {
      Object.keys(data.simulation_data[busIdx]).forEach(tripIdx => {
        allTripKeys.push(`${busIdx}-${tripIdx}`);
      });
    });

    const newSelections: { [key: string]: boolean } = {};
    allTripKeys.forEach(key => { newSelections[key] = isSelected; });
    setSelectedRoutes(newSelections);

    if (isSelected) {
      setRoutesToFetch(allTripKeys);
    }
  };

  const getAlgorithmDisplayName = (algo: string): string => {
    switch (algo) {
      case 'ea2_memetic': return 'EA + Memetic';
      case 'ea2': return 'EA Standard';
      case 'pendelverkehr': return 'Shuttle';
      case 'alns': return 'ALNS (Baseline)';
      default: return algo.toUpperCase();
    }
  };

  const currentSimulationData = data ? data.simulation_data : defaultSimulationData;
  const currentBestSolution = data ? data.best_solution : defaultBestSolution;
  const overallCost = data ? data.overall_cost : 0;

  return (
    <div className="fullscreen-app-container">
      <div className="fullscreen-map">
        <LeafletMap
          locationData={mergedLocationData}
          simulationData={currentSimulationData}
          bestSolution={currentBestSolution}
          selectedRoutes={selectedRoutes}
          numDepots={totalDepotCount}
          defaultDepotsCount={defaultDepotsCount}
          onAddEvacCenter={handleAddEvacCenter}
          onRemoveEvacCenter={handleRemoveEvacCenter}
          onUpdateUserEvacCenter={handleUpdateEvacCenter}
          onSetMainCenter={handleSetMainCenter}
          isSetMainCenterMode={isSetMainCenterMode}
          routesToFetch={routesToFetch}
          placingVehicleId={placingVehicleId}
          onSetVehicleCoordinate={handleSetVehicleCoordinate}
          backendVehicles={data?.vehicles}
          uiVehicles={vehicles}
          customMainCenter={customMainCenter}
          bufferMeters={simulationParams.buffer_meters}
        />
      </div>

      <div className="control-toggles">
        <button
          className={`toggle-button ${showParameters ? 'active' : ''}`}
          onClick={() => setShowParameters(!showParameters)}
        >
          Parameters
        </button>

        <button
          className={`toggle-button ${showRouteControls ? 'active' : ''}`}
          onClick={() => setShowRouteControls(!showRouteControls)}
        >
          Routes
        </button>

        {data && (
          <button
            className={`toggle-button ${showMetrics ? 'active' : ''}`}
            onClick={() => setShowMetrics(!showMetrics)}
          >
            Results
          </button>
        )}

        {data && (
          <button
            className={`toggle-button ${showGanttChart ? 'active' : ''}`}
            onClick={() => setShowGanttChart(!showGanttChart)}
          >
            Timeline
          </button>
        )}
      </div>

      {showParameters && (
        <div ref={parametersRef} className="panel parameters-panel">
          <div className="panel-header">
            <h3>Simulation Parameters</h3>
            <button
              className="panel-close"
              onClick={() => setShowParameters(false)}
            >
              ×
            </button>
          </div>

          <div className="panel-content">
            <SimulationParameters
              onParametersChange={handleParametersChange}
              isRunning={loading}
              onRunSimulation={runEvacuation}
              onSetMainCenterMode={setIsSetMainCenterMode}
              isSetMainCenterMode={isSetMainCenterMode}
              customMainCenterCoords={customMainCenter}
              onClearMainCenter={handleClearMainCenter}
              depots={contextDepots}
              facilities={contextFacilities}
              useCustomFleet={useCustomFleet}
              onUseCustomFleetChange={setUseCustomFleet}
              vehicles={vehicles}
              onVehiclesChange={setVehicles}
              placingVehicleId={placingVehicleId}
              onSetVehiclePlacementMode={setPlacingVehicleId}
              onLoadBombPreset={loadBombEvacuationPreset}
              onLoadFloodPreset={loadWilhelmsburgFloodPreset}
              onLoadDefaultPreset={loadDefaultScenarioPreset}
              currentParams={simulationParams}
            />

            <div className="instructions">
              <p>Click on the map to add evacuation centers.</p>
              <p>Configure parameters, then run the simulation.</p>
            </div>
          </div>
        </div>
      )}

      {showRouteControls && data && (
        <div ref={routeControlsRef} className="panel route-controls-panel">
          <div className="panel-header">
            <h3>Route Controls</h3>
            <button
              className="panel-close"
              onClick={() => setShowRouteControls(false)}
            >
              ×
            </button>
          </div>

          <div className="panel-content scrollable">
            <RouteControlPanel
              simulationData={currentSimulationData}
              bestSolution={currentBestSolution}
              selectedRoutes={selectedRoutes}
              onRouteToggle={handleRouteToggle}
              onBusToggle={handleBusToggle}
              onToggleAll={handleToggleAllRoutes}
            />
          </div>
        </div>
      )}

      {showMetrics && data && (
        <div ref={resultsRef} className="panel results-panel">
          <div className="panel-header">
            <h3>Simulation Results</h3>
            <button
              className="panel-close"
              onClick={() => setShowMetrics(false)}
            >
              ×
            </button>
          </div>

          <div className="panel-content">
            <div className="results-scale-target" style={{ '--results-scale': resultsScale } as React.CSSProperties}>
              {/* Added gap and alignItems to ensure content fits better */}
              <div className="metrics-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
                <div className="metrics-column">
                  {/* FIX: Added inline style to prevent text cutoff */}
                  <p style={{ whiteSpace: 'normal', wordBreak: 'break-word', lineHeight: '1.4' }}>
                    <strong>Total Cost:</strong> {overallCost.toLocaleString()}
                    <br /> {/* Added line break for the badge so it doesn't crowd the number */}
                    <span className={`algorithm-badge ${data.algorithm}`} style={{ display: 'inline-block', marginTop: '4px' }}>
                      {getAlgorithmDisplayName(data.algorithm)}
                    </span>
                  </p>
                  <p>
                    <strong>Vehicles:</strong> {data.num_buses}
                  </p>
                  {(data.algorithm === 'ea2' || data.algorithm === 'ea2_memetic') && (
                    /* FIX: Added inline style to allow wrapping */
                    <p style={{ whiteSpace: 'normal', wordBreak: 'break-word' }}>
                      <strong>Runtime Limit:</strong> {simulationParams.time_limit_seconds} seconds
                    </p>
                  )}
                </div>

                <div className="metrics-column">
                  <div className="metric-item">
                    <strong>Avg Evac Time:</strong> {/* Shortened label slightly to save space */}
                    <span className="metric-value">
                      {evacuationMetrics.averageEvacTime.toFixed(1)} min
                    </span>
                  </div>

                  <div className="metric-item">
                    <strong>Latest Evac:</strong> {/* Shortened label */}
                    <span className={`metric-value ${evacuationMetrics.latestEvacTime > 120 ? 'metric-warning' : 'metric-good'}`}>
                      {evacuationMetrics.latestEvacTime.toFixed(1)} min
                    </span>
                  </div>

                  <div className="metric-item">
                    <strong>Total Wait:</strong> {/* Shortened label */}
                    <span className="metric-value">
                      {evacuationMetrics.cumulativeWaitingTime.toFixed(1)} min
                    </span>
                  </div>

                  <div className="metric-item">
                    <strong>Evacuees:</strong>
                    <span className="metric-value">
                      {evacuationMetrics.totalEvacuees}
                    </span>
                  </div>
                </div>
              </div>

              <div className="action-buttons" style={{ marginTop: '1em' }}>
                <button
                  className="compact-button animation-button"
                  onClick={() => setShowAllRoutes(true)}
                  style={{ width: '100%' }}
                >
                  Show All Routes Summary
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {showGanttChart && data && (
        <div ref={ganttRef} className="panel gantt-panel">
          <div className="panel-header">
            <h3>Timeline Chart</h3>
            <button
              className="panel-close"
              onClick={() => setShowGanttChart(false)}
            >
              ×
            </button>
          </div>

          <div className="panel-content">
            <BusGanttChart
              simulationData={currentSimulationData}
              vehicles={data.vehicles}
              bestSolution={currentBestSolution}
              locationData={mergedLocationData}
              numDepots={totalDepotCount}
            />
          </div>

          <div className="resize-handle"></div>
        </div>
      )}

      {loading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <div className="loading-text">Running Simulation...</div>
        </div>
      )}

      {showAllRoutes && data && (
        <AllRoutesView
          simulationData={currentSimulationData}
          bestSolution={currentBestSolution}
          locationData={mergedLocationData}
          numDepots={totalDepotCount}
          vehicles={data.vehicles}
          onClose={() => setShowAllRoutes(false)}
        />
      )}
    </div>
  );
};

export default FullscreenMapApp;
