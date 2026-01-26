import type { Metadata } from 'next';
import './globals.css';
import { EvacuationProvider } from './context/EvacuationContext';
import { Suspense } from 'react';
import Script from 'next/script';

export const metadata: Metadata = {
  title: 'Evacuation Route Planning',
  description: 'Vehicle-Based Evacuation Planning Application',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        {/* Load Leaflet CSS directly from CDN */}
        <link
          rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
          crossOrigin=""
        />
      </head>
      <body>
        <EvacuationProvider>
          {/* Remove the Suspense boundary from here */}
          {children}
        </EvacuationProvider>
        
        {/* Add these scripts to handle hydration errors more gracefully */}
        <Script id="hydration-handler" strategy="afterInteractive">
          {`
            // This helps suppress hydration errors in production
            window.__NEXT_HYDRATION_MARK_END__ = window.__NEXT_HYDRATION_MARK_END__ || function(){}
          `}
        </Script>
      </body>
    </html>
  );
}
