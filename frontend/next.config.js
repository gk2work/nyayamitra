/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy API calls to FastAPI backend during development
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8080/api/:path*",
      },
    ];
  },
};

module.exports = nextConfig;
