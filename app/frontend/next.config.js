/** @type {import('next').NextConfig} */
const fs = require('fs');

const isDocker = (() => {
  try {
    return fs.existsSync('/.dockerenv');
  } catch {
    return false;
  }
})();

const backendOrigin = (process.env.BACKEND_ORIGIN ||
  (isDocker ? 'http://backend:8000' : 'http://localhost:8000')
).replace(/\/$/, '');

const nextConfig = {
  reactStrictMode: false, // Disable strict mode to reduce double-rendering issues
  swcMinify: true,
  // Enable static exports if you need to deploy to a static host
  // output: 'export',
  output: 'standalone',
    // This will prevent the build from failing on ESLint errors.
  // You will still see warnings/errors in your local development console.
  eslint: {
    ignoreDuringBuilds: true,
  },
  // Configure Webpack to handle CSS from the old React app
  webpack: (config) => {
    config.resolve.fallback = { fs: false };
    return config;
  },
  
  // Configure images domain if you're loading external images
  images: {
    domains: ['unpkg.com', 'maps.google.com'],
  },
  
  // Ensure rewrites are handled correctly
  async rewrites() {
    return [
      // Proxy API requests to your backend
      {
        source: '/api/:path*',
        destination: `${backendOrigin}/:path*`,
      },
    ];
  },
  
  // This helps with Leaflet and other libraries that need window access
  experimental: {
    esmExternals: 'loose',
  },
};

module.exports = nextConfig;
