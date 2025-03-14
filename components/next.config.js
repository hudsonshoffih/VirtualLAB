/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    eslint: {
      // Warning: This allows production builds to successfully complete even with ESLint errors
      ignoreDuringBuilds: true,
    },
    typescript: {
      // Warning: This allows production builds to successfully complete even with TypeScript errors
      ignoreBuildErrors: true,
    },
    images: {
      domains: ["your-domain.com"], // Add any domains you need for external images
    },
  }
  
  module.exports = nextConfig
  
  