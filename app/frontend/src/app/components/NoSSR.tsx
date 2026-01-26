'use client';

import React, { useState, useEffect } from 'react';

// This is a wrapper component that prevents server-side rendering of its children
const NoSSR: React.FC<{ 
  children: React.ReactNode, 
  fallback?: React.ReactNode 
}> = ({ children, fallback = null }) => {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  // On the server, return the fallback
  // Once on the client, render the children
  return isClient ? <>{children}</> : <>{fallback}</>;
};

export default NoSSR;