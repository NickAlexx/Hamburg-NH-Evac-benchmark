'use client';

import dynamic from 'next/dynamic';

// Import the new CSS file
import '../styles/App.css';
import '../styles/FullscreenApp.css';

// We need to dynamically load the App component because Leaflet requires the window object
const EvacuationApp = dynamic(() => import('./components/FullscreenMapApp'), {
  ssr: false,
  loading: () => <div className="loading-container">Loading Evacuation Planning Interface...</div>
});

export default function Home() {
  return (
    <main>
      <EvacuationApp />
    </main>
  );
}