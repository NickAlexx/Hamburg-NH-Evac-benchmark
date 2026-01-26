// Path: frontend2/src/app/components/LeafletMap.tsx
'use client';

import React, { useEffect, useState, useRef, useMemo, useCallback } from 'react';
import L from 'leaflet';
import {
  MapContainer,
  TileLayer,
  Marker,
  Circle,
  Polyline,
  useMapEvents,
  useMap,
  Popup
} from 'react-leaflet';
import axios from 'axios';
import { Trip, BackendVehicle } from './types';
import dynamic from 'next/dynamic';
import { BOMB_EVACUATION_PRESET } from '../data/scenarios';

// Dynamically import components that depend on window object
const FacilityMarker = dynamic(() => import('./FacilityMarker'), { ssr: false });

// NEW CONSTANT: Read the API base URL from environment variables
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;

// *** NEW TYPE DEFINITION for UI vehicle state ***
export interface Vehicle {
  id: string;
  capacity: number;
  start: 
    | { kind: 'depot', index: number }
    | { kind: 'node', index: number }
    | { kind: 'coord', coords: [number, number] | null };
}

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

export interface UserEvacCenter {
  label: string;
  coords: [number, number]; // (lon, lat)
  capacity?: number | null;
}

interface LeafletMapProps {
  locationData: Location[];
  simulationData: SimulationData;
  bestSolution: Trip[][];
  selectedRoutes: { [key: string]: boolean };
  numDepots: number;
  defaultDepotsCount?: number;
  onAddEvacCenter: (center: UserEvacCenter) => void;
  onRemoveEvacCenter?: (index: number) => void;
  onUpdateUserEvacCenter: (index: number, centerUpdate: Partial<UserEvacCenter>) => void;
  onSetMainCenter: (coords: [number, number]) => void;
  isSetMainCenterMode: boolean;
  onRoutePathsUpdate?: (paths: { [key: string]: [number, number][] }) => void;
  disableRouteLoading?: boolean;
  initialRoutePaths?: { [key: string]: [number, number][] };
  routesToFetch?: string[];
  placingVehicleId: string | null;
  onSetVehicleCoordinate: (vehicleId: string, coords: [number, number]) => void;
  backendVehicles?: BackendVehicle[];
  uiVehicles?: Vehicle[];
  customMainCenter?: [number, number] | null;
  bufferMeters: number;
}

const DEFAULT_DEPOT_ICON = {
  url: 'https://unpkg.com/leaflet@1.9.3/dist/images/marker-icon.png',
  size: [25, 41] as [number, number],
  anchor: [12, 41] as [number, number],
  popupAnchor: [1, -34] as [number, number],
};

// *** NEW: Bus colors for routes, consistent with Gantt chart ***
const busColors = [
  '#e41a1c', '#377eb8', '#ff7f00', '#984ea3', '#4daf4a',
  '#f781bf', '#a65628', '#66c2a5', '#7570b3', '#e6ab02',
];

// *** NEW ICON for custom vehicle start points ***
const vehicleStartIcon = (label: string, count: number) => new L.DivIcon({
  className: '',
  iconSize: [36, 52],
  iconAnchor: [18, 52],
  html: `<div class="vehicle-start-marker">
           <span class="vehicle-start-icon">🚌</span>
           <span class="vehicle-start-label">${label}</span>
           ${count > 1 ? `<span class="vehicle-start-count">${count}</span>` : ''}
         </div>`
});

const bufferCenterIcon = new L.Icon({
  iconUrl: 'http://maps.google.com/mapfiles/ms/icons/yellow-dot.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.3/dist/images/marker-shadow.png',
  iconSize: [32, 32],
  iconAnchor: [16, 32],
  popupAnchor: [0, -32]
});

const DEPOT_LABEL_HEIGHT_PX = 16;
const VEHICLE_START_GROUP_PRECISION = 6;

const formatVehicleGroupLabel = (indices: number[]): string => {
  const sorted = [...indices].sort((a, b) => a - b);
  if (sorted.length === 1) return `V${sorted[0]}`;
  if (sorted.length <= 3) return sorted.map(index => `V${index}`).join(', ');
  const first = sorted[0];
  const last = sorted[sorted.length - 1];
  const isContiguous = sorted.every((value, idx) => idx === 0 || value === sorted[idx - 1] + 1);
  return isContiguous ? `V${first}-${last}` : `V${first}..${last}`;
};

const bombIcon = new L.DivIcon({
  className: '',
  iconSize: [28, 28],
  iconAnchor: [14, 14],
  popupAnchor: [0, -14],
  html: `
    <svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
      <circle cx="14" cy="14" r="8" fill="#111" stroke="#f5f5f5" stroke-width="2" />
      <path d="M18 6 L24 4" stroke="#c0392b" stroke-width="2" stroke-linecap="round" />
      <circle cx="24" cy="4" r="2" fill="#f1c40f" />
      <circle cx="11" cy="11" r="2" fill="rgba(255,255,255,0.2)" />
    </svg>
  `
});

const USER_DEPOT_ICON_URLS = [
  'http://maps.google.com/mapfiles/ms/icons/red-dot.png',
  'http://maps.google.com/mapfiles/ms/icons/blue-dot.png',
  'http://maps.google.com/mapfiles/ms/icons/green-dot.png',
  'http://maps.google.com/mapfiles/ms/icons/orange-dot.png',
  'http://maps.google.com/mapfiles/ms/icons/purple-dot.png',
];
const USER_DEPOT_ICON_SIZE: [number, number] = [32, 32];
const USER_DEPOT_ICON_ANCHOR: [number, number] = [16, 32];
const USER_DEPOT_POPUP_ANCHOR: [number, number] = [0, -32];

const createDepotIcon = (
  iconUrl: string,
  iconSize: [number, number],
  iconAnchor: [number, number],
  popupAnchor: [number, number],
  capacityBadge: string
) => new L.DivIcon({
  className: '',
  iconSize: [iconSize[0], iconSize[1] + DEPOT_LABEL_HEIGHT_PX],
  iconAnchor,
  popupAnchor,
  html: `
    <div class="depot-marker">
      <img class="depot-marker-icon" src="${iconUrl}" alt="" style="width: ${iconSize[0]}px; height: ${iconSize[1]}px;" />
      <div class="depot-marker-cap">${capacityBadge}</div>
    </div>
  `
});

function MapClickHandler({
  onAddEvacCenter,
  onSetMainCenter,
  isSetMainCenterMode,
  placingVehicleId,
  onSetVehicleCoordinate,
  isAddDepotMode,
}: {
  onAddEvacCenter: (center: UserEvacCenter) => void;
  onSetMainCenter: (coords: [number, number]) => void;
  isSetMainCenterMode: boolean;
  placingVehicleId: string | null;
  onSetVehicleCoordinate: (vehicleId: string, coords: [number, number]) => void;
  isAddDepotMode: boolean;
}) {
  const map = useMap();

  useMapEvents({
    click(e: L.LeafletMouseEvent) {
      if (placingVehicleId) {
        onSetVehicleCoordinate(placingVehicleId, [e.latlng.lng, e.latlng.lat]);
      } else if (isSetMainCenterMode) {
        onSetMainCenter([e.latlng.lng, e.latlng.lat]);
      } else if (isAddDepotMode) {
        // Only add depot if explicitly in Add Depot Mode
        onAddEvacCenter({ label: "User Depot", coords: [e.latlng.lng, e.latlng.lat] });
      }
    },
  });

  useEffect(() => {
    if (placingVehicleId || isSetMainCenterMode || isAddDepotMode) {
      map.getContainer().style.cursor = 'crosshair';
    } else {
      map.getContainer().style.cursor = '';
    }
  }, [isSetMainCenterMode, placingVehicleId, isAddDepotMode, map]);

  return null;
}

function MapScaleControl() {
  const map = useMap();

  useEffect(() => {
    const scaleControl = L.control.scale({
      metric: true,
      imperial: false,
      maxWidth: 120,
      position: 'bottomright',
    });

    scaleControl.addTo(map);

    return () => {
      scaleControl.remove();
    };
  }, [map]);

  return null;
}

function createDirectPath(coords: [number, number][]): [number, number][] {
  if (coords.length < 2) return coords;
  
  const result: [number, number][] = [];
  result.push(coords[0]);
  
  for (let i = 0; i < coords.length - 1; i++) {
    const start = coords[i];
    const end = coords[i + 1];
    
    const pointsToAdd = 10;
    for (let j = 1; j <= pointsToAdd; j++) {
      const fraction = j / (pointsToAdd + 1);
      const lon = start[0] + (end[0] - start[0]) * fraction;
      const lat = start[1] + (end[1] - start[1]) * fraction;
      result.push([lon, lat]);
    }
    
    result.push(end);
  }
  
  return result;
}

function useThrottledFetch(delay = 500) {
  const lastFetchTime = useRef<number>(0);
  const pendingFetches = useRef<Map<string, Promise<[number, number][]>>>(new Map());
  
  return useCallback(async (key: string, fetchFn: () => Promise<[number, number][]>) => {
    if (pendingFetches.current.has(key)) {
      const pendingPromise = pendingFetches.current.get(key);
      if (pendingPromise) {
        return pendingPromise;
      }
    }
    
    const now = Date.now();
    const timeSinceLastFetch = now - lastFetchTime.current;
    
    if (timeSinceLastFetch < delay) {
      await new Promise(resolve => setTimeout(resolve, delay - timeSinceLastFetch));
    }
    
    lastFetchTime.current = Date.now();
    const fetchPromise = fetchFn();
    pendingFetches.current.set(key, fetchPromise);
    
    fetchPromise.finally(() => {
      pendingFetches.current.delete(key);
    });
    
    return fetchPromise;
  }, [delay]);
}

const LeafletMap: React.FC<LeafletMapProps> = ({
  locationData,
  simulationData,
  bestSolution,
  selectedRoutes,
  numDepots,
  defaultDepotsCount = 1,
  onAddEvacCenter,
  onRemoveEvacCenter,
  onUpdateUserEvacCenter,
  onSetMainCenter,
  isSetMainCenterMode,
  disableRouteLoading = false,
  initialRoutePaths = {},
  routesToFetch = [],
  placingVehicleId,
  onSetVehicleCoordinate,
  backendVehicles,
  uiVehicles,
  customMainCenter,
  bufferMeters,
}) => {
  const [routePaths, setRoutePaths] = useState<{ [key: string]: [number, number][] }>(initialRoutePaths);
  const [isAddDepotMode, setIsAddDepotMode] = useState<boolean>(false);
  
  const routeCache = useRef<{ [key: string]: [number, number][] }>(initialRoutePaths);
  const routeLoadingInProgress = useRef<Set<string>>(new Set());
  
  const simulationDataRef = useRef(simulationData);
  const bestSolutionRef = useRef(bestSolution);
  
  const throttledFetch = useThrottledFetch(500);

  // KEY FIX: Clean up on unmount
  useEffect(() => {
    const inProgress = routeLoadingInProgress.current;
    return () => {
      // Clear any pending operations on unmount
      inProgress.clear();
    };
  }, []);

  useEffect(() => {
    const isNewSimulation = (
      simulationData !== simulationDataRef.current ||
      bestSolution !== bestSolutionRef.current
    );
    
    if (isNewSimulation) {
      console.log('New simulation detected, clearing route cache...');
      routeCache.current = {};
      routeLoadingInProgress.current.clear();
      setRoutePaths({});
      
      simulationDataRef.current = simulationData;
      bestSolutionRef.current = bestSolution;
    }
  }, [simulationData, bestSolution]);

  const fetchORSRoute = useCallback(async (key: string, coords: [number, number][]): Promise<[number, number][]> => {
    if (routeLoadingInProgress.current.has(key)) {
        return createDirectPath(coords);
    }

    if (coords.length < 2 || (coords.length === 2 && coords[0][0] === coords[1][0] && coords[0][1] === coords[1][1])) {
        console.log(`Skipping ORS fetch for degenerate route: ${key}`);
        return createDirectPath(coords);
    }

    routeLoadingInProgress.current.add(key);

    try {
        return await throttledFetch(key, async () => {
            const maxRetries = 3;
            const retryDelay = 1000;

            for (let attempt = 1; attempt <= maxRetries; attempt++) {
                try {
                    const response = await axios.post(`${API_BASE_URL}/ors-proxy`, {
                        coordinates: coords
                    });
                    const routeCoords = response.data.features[0].geometry.coordinates;
                    return routeCoords.length > 20 ? routeCoords : createDirectPath(coords);
                } catch (error: unknown) {
                    const isAxiosError = axios.isAxiosError(error);
                    const isRetryable = isAxiosError && error.response && [404, 500, 502, 503, 504].includes(error.response.status);

                    if (isRetryable && attempt < maxRetries) {
                        console.warn(`ORS request for route ${key} failed (attempt ${attempt}/${maxRetries}). Retrying in ${retryDelay}ms...`);
                        await new Promise(resolve => setTimeout(resolve, retryDelay));
                    } else {
                        console.error(`Final attempt for route ${key} failed.`);
                        throw error;
                    }
                }
            }
            throw new Error("Retry loop finished without success or error.");
        });
    } catch {
        console.error(`Error fetching route from ORS for key ${key} after all retries. Falling back to direct path.`);
        return createDirectPath(coords);
    } finally {
        routeLoadingInProgress.current.delete(key);
    }
  }, [throttledFetch]);

  useEffect(() => {
    if (disableRouteLoading || routesToFetch.length === 0) {
      return;
    }

    const fetchQueuedRoutes = async () => {
      const keysThatNeedFetching = routesToFetch.filter(key =>
        !routeCache.current[key] && !routeLoadingInProgress.current.has(key)
      );

      if (keysThatNeedFetching.length === 0) {
        return;
      }

      console.log(`On-demand fetch triggered for ${keysThatNeedFetching.length} routes.`);

      const fetchPromises = keysThatNeedFetching.map(async (key) => {
        const [busIdxStr, tripIdxStr] = key.split('-');
        const busIdx = parseInt(busIdxStr, 10);
        const tripIdx = parseInt(tripIdxStr, 10);

        const tripObject = bestSolution[busIdx]?.[tripIdx];
        if (!tripObject) return;

        const routeCoords: [number, number][] = [];
        
        const vehicle = backendVehicles?.[busIdx];
        if (tripIdx === 0 && vehicle?.start?.kind === 'coord' && vehicle.start.lat && vehicle.start.lon) {
          routeCoords.push([vehicle.start.lon, vehicle.start.lat]);
        } else {
          if (locationData[tripObject.start_depot]) {
              routeCoords.push(locationData[tripObject.start_depot].coords);
          }
        }

        tripObject.stops.forEach(fIdx => {
            const facLoc = locationData[numDepots + fIdx];
            if (facLoc) routeCoords.push(facLoc.coords);
        });
        if (locationData[tripObject.end_depot]) {
            routeCoords.push(locationData[tripObject.end_depot].coords);
        }

        if (routeCoords.length >= 2) {
          try {
            const path = await fetchORSRoute(key, routeCoords);
            routeCache.current[key] = path;
            setRoutePaths(prev => ({ ...prev, [key]: path }));
          } catch {
            const directPath = createDirectPath(routeCoords);
            routeCache.current[key] = directPath;
            setRoutePaths(prev => ({ ...prev, [key]: directPath }));
          }
        }
      });

      await Promise.all(fetchPromises);
    };

    fetchQueuedRoutes();
  }, [routesToFetch, bestSolution, locationData, numDepots, fetchORSRoute, disableRouteLoading, backendVehicles]);

  const initialCenter: [number, number] = locationData.length
    ? [locationData[0].coords[1], locationData[0].coords[0]]
    : [53.55, 9.95];

  const customMainCenterPosition: [number, number] | null = customMainCenter
    ? [customMainCenter[1], customMainCenter[0]]
    : null;

  const isBombScenario = Math.round(bufferMeters) === Math.round(BOMB_EVACUATION_PRESET.bufferMeters);

  const groupedVehicleStarts = useMemo(() => {
    if (!uiVehicles) return [];
    const groups = new Map<string, { coords: [number, number]; entries: { vehicle: Vehicle; index: number }[] }>();

    uiVehicles.forEach((vehicle, index) => {
      if (vehicle.start.kind !== 'coord' || !vehicle.start.coords) return;
      const [lon, lat] = vehicle.start.coords;
      const key = `${lon.toFixed(VEHICLE_START_GROUP_PRECISION)}|${lat.toFixed(VEHICLE_START_GROUP_PRECISION)}`;
      const group = groups.get(key);
      if (group) {
        group.entries.push({ vehicle, index });
      } else {
        groups.set(key, { coords: vehicle.start.coords, entries: [{ vehicle, index }] });
      }
    });

    return Array.from(groups.entries()).map(([key, group]) => ({ key, ...group }));
  }, [uiVehicles]);
    
  const displayedRoutes = useMemo(() => {
    return Object.keys(routePaths)
      .filter(key => selectedRoutes[key])
      .reduce((acc, key) => {
        acc[key] = routePaths[key];
        return acc;
      }, {} as { [key: string]: [number, number][] });
  }, [routePaths, selectedRoutes]);

  return (
    <div className="leaflet-map-wrapper" style={{ width: '100%', height: '100%', position: 'relative' }}>
      <MapContainer
        center={initialCenter} 
        zoom={12} 
        zoomSnap={0.25}
        zoomDelta={0.25}
        style={{ width: '100%', height: '100%' }} 
        className="leaflet-container-responsive"
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution="© OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <MapScaleControl />

        {customMainCenterPosition && (
          <>
            {isBombScenario && bufferMeters > 0 && (
              <Circle
                center={customMainCenterPosition}
                radius={bufferMeters}
                pathOptions={{
                  color: '#e74c3c',
                  fillColor: '#e74c3c',
                  fillOpacity: 0.15,
                  weight: 2
                }}
                interactive={false}
              />
            )}
            <Marker 
              position={customMainCenterPosition} 
              icon={isBombScenario ? bombIcon : bufferCenterIcon}
              zIndexOffset={-100}
            >
              <Popup>
                <strong>{isBombScenario ? 'Bomb Location' : 'Buffer Center'}</strong><br />
                {isBombScenario ? (
                  <>Exclusion zone radius: {bufferMeters} m</>
                ) : (
                  <>This point defines the evacuation area for facilities. It is not a destination.</>
                )}
              </Popup>
            </Marker>
          </>
        )}

        <MapClickHandler
          onAddEvacCenter={onAddEvacCenter}
          onSetMainCenter={onSetMainCenter}
          isSetMainCenterMode={isSetMainCenterMode}
          placingVehicleId={placingVehicleId}
          onSetVehicleCoordinate={onSetVehicleCoordinate}
          isAddDepotMode={isAddDepotMode}
        />

        {groupedVehicleStarts.map(group => {
          const position: [number, number] = [group.coords[1], group.coords[0]];
          const sortedEntries = [...group.entries].sort((a, b) => a.index - b.index);
          const indices = sortedEntries.map(entry => entry.index);
          const label = formatVehicleGroupLabel(indices);
          const icon = vehicleStartIcon(label, indices.length);
          const isSingle = indices.length === 1;

          return (
            <Marker key={`vehicle-start-${group.key}`} position={position} icon={icon} zIndexOffset={500}>
              <Popup>
                <strong>{isSingle ? `Vehicle ${indices[0]} Start` : `Vehicle Starts (${indices.length})`}</strong><br />
                {isSingle ? (
                  <>
                    Capacity: {sortedEntries[0].vehicle.capacity}<br />
                    Custom Coordinate
                  </>
                ) : (
                  <div style={{ marginTop: '6px' }}>
                    {sortedEntries.map(({ vehicle, index }) => (
                      <div key={vehicle.id}>
                        V{index}: Cap {vehicle.capacity}
                      </div>
                    ))}
                  </div>
                )}
              </Popup>
            </Marker>
          );
        })}

        {locationData.slice(0, numDepots).map((loc, idx) => {
          const markerPos: [number, number] = [loc.coords[1], loc.coords[0]];
          const capacityValue = loc.capacity == null ? null : loc.capacity;
          const capacityLabel = capacityValue == null ? 'Unlimited' : capacityValue.toLocaleString();
          const capacityBadge = capacityValue == null ? 'Cap: Unlimited' : `Cap: ${capacityLabel}`;

          if (idx < defaultDepotsCount) {
            const icon = createDepotIcon(
              DEFAULT_DEPOT_ICON.url,
              DEFAULT_DEPOT_ICON.size,
              DEFAULT_DEPOT_ICON.anchor,
              DEFAULT_DEPOT_ICON.popupAnchor,
              capacityBadge
            );
            return (
              <Marker key={`depot-${idx}`} position={markerPos} icon={icon}>
                <Popup>
                  <strong>{loc.label}</strong>
                  <br />
                  (Default Depot)
                  <br />
                  Capacity: {capacityLabel}
                </Popup>
              </Marker>
            );
          } else {
            const userCenterIndex = idx - defaultDepotsCount;
            const iconUrl = USER_DEPOT_ICON_URLS[userCenterIndex % USER_DEPOT_ICON_URLS.length];
            const icon = createDepotIcon(
              iconUrl,
              USER_DEPOT_ICON_SIZE,
              USER_DEPOT_ICON_ANCHOR,
              USER_DEPOT_POPUP_ANCHOR,
              capacityBadge
            );
            return (
              <Marker key={`depot-${idx}`} position={markerPos} icon={icon}>
                <Popup>
                  <strong>{loc.label}</strong>
                  <br />
                  (User Depot)
                  <br />
                  Capacity: {capacityLabel}
                  <br />
                  <div style={{ marginTop: '8px' }}>
                    <label htmlFor={`capacity-${idx}`} style={{ fontSize: '12px', display: 'block', marginBottom: '4px' }}>
                      Capacity:
                    </label>
                    <input
                      id={`capacity-${idx}`}
                      type="number"
                      placeholder="Infinite"
                      defaultValue={loc.capacity || ''}
                      onBlur={(e) => { // Using onBlur to avoid updating on every keystroke
                        const newCapacity = e.target.value ? parseInt(e.target.value, 10) : null;
                        if (onUpdateUserEvacCenter) {
                          onUpdateUserEvacCenter(userCenterIndex, { capacity: newCapacity });
                        }
                      }}
                      onClick={(e) => e.stopPropagation()} // Prevent map click event
                      style={{ width: '100px', padding: '4px', fontSize: '12px', border: '1px solid #ccc', borderRadius: '4px' }}
                    />
                  </div>
                  <button 
                    onClick={(e) => {
                      e.stopPropagation();
                      if (onRemoveEvacCenter) {
                        onRemoveEvacCenter(userCenterIndex);
                      }
                    }}
                    style={{
                      marginTop: '8px',
                      backgroundColor: '#e74c3c',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      padding: '4px 8px',
                      cursor: 'pointer',
                      fontSize: '12px'
                    }}
                  >
                    Remove Depot
                  </button>
                </Popup>
              </Marker>
            );
          }
        })}

        {locationData.slice(numDepots).map((loc, idx) => {
          const markerPos: [number, number] = [loc.coords[1], loc.coords[0]];
          const facilityId = idx;
          
          return (
            <FacilityMarker
              key={`facility-${facilityId}`}
              position={markerPos}
              label={loc.label}
              initialEvacuees={loc.people}
              remainingEvacuees={loc.people}
              facilityId={facilityId}
            />
          );
        })}

        {Object.entries(displayedRoutes).map(([routeKey, coords]) => {
          if (!coords || coords.length < 2) return null;
          const latLngs = coords.map(([lon, lat]) => [lat, lon] as [number, number]);
          
          const [busIdxStr, tripIdxStr] = routeKey.split('-');
          const busIndex = parseInt(busIdxStr, 10);
          const tripIndex = parseInt(tripIdxStr, 10);
          const routeColor = busColors[busIndex % busColors.length];

          // Data retrieval for popup
          const tripObject = bestSolution[busIndex]?.[tripIndex];
          const tripSim = simulationData[busIdxStr]?.[tripIdxStr];
          const vehicle = backendVehicles?.[busIndex];
          const capacity = vehicle?.capacity ?? 'N/A';

          let popupContent = (
             <div>
                <strong>Vehicle {busIndex}</strong> (Trip {tripIndex})
             </div>
          );

          if (tripObject && tripSim) {
              let startLabel = `Depot ${tripObject.start_depot}`;
              // Handle custom start coordinate case for the first trip
              if (tripIndex === 0 && vehicle?.start?.kind === 'coord') {
                  startLabel = 'Custom Coord';
              } else if (locationData[tripObject.start_depot]) {
                  startLabel = locationData[tripObject.start_depot].label.split(',')[0];
              }

              const endLabel = locationData[tripObject.end_depot]?.label?.split(',')[0] || `Depot ${tripObject.end_depot}`;
              
              // Calculate total evacuees picked up
              const evacueesCount = tripSim.details.reduce((acc, detail) => {
                const match = detail.match(/picked up (\d+)/);
                return match ? acc + parseInt(match[1], 10) : acc;
              }, 0);

              // Get list of stops names
              const stopsList = tripObject.stops.map(stopIdx => {
                const loc = locationData[numDepots + stopIdx];
                return loc ? loc.label.split(',')[0] : `Facility ${stopIdx}`;
              });

              popupContent = (
                  <div style={{ fontSize: '13px', lineHeight: '1.5', minWidth: '220px' }}>
                      <div style={{ 
                        marginBottom: '8px', 
                        borderBottom: `2px solid ${routeColor}`, 
                        paddingBottom: '4px',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center'
                      }}>
                        <strong>Vehicle {busIndex}</strong> 
                        <span style={{
                          backgroundColor: '#eee', 
                          padding: '1px 6px', 
                          borderRadius: '4px', 
                          fontSize: '0.9em',
                          color: '#555'
                        }}>Cap: {capacity}</span>
                      </div>
                      
                      <div style={{ display: 'grid', gridTemplateColumns: '20px 1fr', gap: '4px', alignItems: 'start' }}>
                        <span>🆔</span> <strong>Trip {tripIndex}</strong>
                        
                        <span>⏱</span> <span>{tripSim.departure.toFixed(1)} - {tripSim.return.toFixed(1)} min</span>
                        
                        <span>👥</span> <span><strong>{evacueesCount}</strong> evacuees</span>
                        
                        <span>🏁</span> 
                        <div style={{ fontSize: '0.95em' }}>
                            <div style={{color: '#555'}}>Start: {startLabel}</div>
                            <div style={{color: '#555'}}>End: {endLabel}</div>
                        </div>
                      </div>

                      {stopsList.length > 0 && (
                        <div style={{ marginTop: '10px', paddingTop: '8px', borderTop: '1px dashed #ccc' }}>
                            <div style={{fontWeight: 'bold', marginBottom: '4px', color: '#333'}}>
                              🛑 Stops ({stopsList.length}):
                            </div>
                            <ul style={{ 
                              margin: 0, 
                              paddingLeft: '18px', 
                              maxHeight: '120px', 
                              overflowY: 'auto',
                              fontSize: '0.95em'
                            }}>
                                {stopsList.map((name, i) => (
                                    <li key={i} style={{marginBottom: '2px'}}>{name}</li>
                                ))}
                            </ul>
                        </div>
                      )}
                  </div>
              );
          }

          return (
            <Polyline 
              key={`route-${routeKey}`} 
              positions={latLngs} 
              pathOptions={{ color: routeColor, weight: 6, opacity: 0.8, lineCap: 'round' }} 
              className="animated-route-path"
            >
                <Popup maxWidth={300}>
                    {popupContent}
                </Popup>
            </Polyline>
          );
        })}
        
      </MapContainer>

      {/* Floating Action Button for Add Depot Mode */}
      <div 
        title={isAddDepotMode ? "Exit Add Depot Mode" : "Add Depot Mode"}
        onClick={(e) => {
          e.stopPropagation(); // Prevent map click when clicking button
          setIsAddDepotMode(!isAddDepotMode);
        }}
        style={{
          position: 'absolute',
          bottom: '30px', 
          left: '20px',
          zIndex: 1000, 
          backgroundColor: isAddDepotMode ? '#e74c3c' : 'white', 
          color: isAddDepotMode ? 'white' : '#333',
          padding: '10px 15px',
          borderRadius: '30px', // Pill shape
          border: '2px solid rgba(0,0,0,0.1)',
          boxShadow: '0 4px 6px rgba(0,0,0,0.2)',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          cursor: 'pointer',
          fontWeight: 'bold',
          fontSize: '14px',
          transition: 'all 0.2s ease',
        }}
      >
        <span style={{ fontSize: '18px', lineHeight: 1 }}>
          {isAddDepotMode ? '✕' : '+'}
        </span>
        <span>
          {isAddDepotMode ? 'Stop Adding Depots' : 'Add Depot'}
        </span>
      </div>
    </div>
  );
};

export default LeafletMap;
