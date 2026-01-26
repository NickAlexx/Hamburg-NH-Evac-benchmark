/** @type {import('postcss-load-config').Config} */
const config = {
  plugins: {
    // Remove tailwindcss reference
    autoprefixer: {},
  },
};

export default config;