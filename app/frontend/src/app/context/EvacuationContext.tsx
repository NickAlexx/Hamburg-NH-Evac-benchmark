// frontend2/src/app/context/EvacuationContext.tsx
'use client';

import React, { createContext, useContext, useState, ReactNode, useCallback } from 'react';
import { Trip } from '../components/types';

interface Location {
  label: string;
  coords: [number, number]; // (lon, lat)
  people: number;
}

// Define UserEvacCenter interface for the context
interface UserEvacCenter {
  label: string;
  coords: [number, number]; // (lon, lat)
  capacity?: number | null;
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

interface EvacuationContextType {
  simulationData: SimulationData;
  bestSolution: Trip[][];
  depots: Location[];
  facilities: Location[];
  userEvacCenters: UserEvacCenter[];
  routePaths: { [key: string]: [number, number][] };
  setSimulationData: (data: SimulationData) => void;
  setBestSolution: (solution: Trip[][]) => void;
  setDepots: (depots: Location[]) => void;
  setFacilities: (facilities: Location[]) => void;
  setUserEvacCenters: (centers: UserEvacCenter[]) => void;
  addUserEvacCenter: (center: UserEvacCenter) => void;
  removeUserEvacCenter: (index: number) => void;
  updateUserEvacCenter: (index: number, centerUpdate: Partial<UserEvacCenter>) => void;
  setRoutePaths: (paths: { [key: string]: [number, number][] }) => void;
}

const EvacuationContext = createContext<EvacuationContextType | undefined>(undefined);

export function EvacuationProvider({ children }: { children: ReactNode }) {
  // Initialize state directly with default empty values. No localStorage needed.
  const [simulationData, setSimulationDataState] = useState<SimulationData>({});
  const [bestSolution, setBestSolutionState] = useState<Trip[][]>([]);
  const [depots, setDepotsState] = useState<Location[]>([]);
  const [facilities, setFacilitiesState] = useState<Location[]>([]);
  const [userEvacCenters, setUserEvacCentersState] = useState<UserEvacCenter[]>([]);
  const [routePaths, setRoutePathsState] = useState<{ [key: string]: [number, number][] }>({});

  // Simplified setters that just update the state.
  const setSimulationData = useCallback((data: SimulationData) => {
    setSimulationDataState(data);
  }, []);

  const setBestSolution = useCallback((solution: Trip[][]) => {
    setBestSolutionState(solution);
  }, []);

  const setDepots = useCallback((newDepots: Location[]) => {
    setDepotsState(newDepots);
  }, []);

  const setFacilities = useCallback((newFacilities: Location[]) => {
    setFacilitiesState(newFacilities);
  }, []);

  const setUserEvacCenters = useCallback((centers: UserEvacCenter[]) => {
    setUserEvacCentersState(centers);
    // Clear old simulation depots when inputs change
    setDepotsState([]);
  }, []);

  const addUserEvacCenter = useCallback((center: UserEvacCenter) => {
    setUserEvacCentersState(prevCenters => [...prevCenters, center]);
    // Clear old simulation depots when inputs change
    setDepotsState([]);
  }, []);

  const removeUserEvacCenter = useCallback((index: number) => {
    setUserEvacCentersState(prevCenters => prevCenters.filter((_, i) => i !== index));
    // Clear old simulation depots when inputs change
    setDepotsState([]);
  }, []);

  const updateUserEvacCenter = useCallback((index: number, centerUpdate: Partial<UserEvacCenter>) => {
    setUserEvacCentersState(prevCenters =>
      prevCenters.map((center, i) =>
        i === index ? { ...center, ...centerUpdate } : center
      )
    );
  }, []);

  const setRoutePaths = useCallback((paths: { [key: string]: [number, number][] }) => {
    setRoutePathsState(paths);
  }, []);

  return (
    <EvacuationContext.Provider
      value={{
        simulationData,
        bestSolution,
        depots,
        facilities,
        userEvacCenters,
        routePaths,
        setSimulationData,
        setBestSolution,
        setDepots,
        setFacilities,
        setUserEvacCenters,
        addUserEvacCenter,
        removeUserEvacCenter,
        updateUserEvacCenter,
        setRoutePaths
      }}
    >
      {children}
    </EvacuationContext.Provider>
  );
}

export function useEvacuation() {
  const context = useContext(EvacuationContext);
  if (context === undefined) {
    throw new Error('useEvacuation must be used within an EvacuationProvider');
  }
  return context;
}