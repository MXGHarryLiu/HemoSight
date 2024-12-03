import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

//const isDocker = import.meta.env.DOCKER === 'true';
//const FastAPIURL = isDocker ? 'http://hematology-fastapi:80' : 'http://localhost:4001';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 3000,
    watch: {
      usePolling: true,
    },
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://hematology-fastapi:80',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      '/ws': {
        target: 'ws://hematology-fastapi:80',
        changeOrigin: true,
        secure: false,
        ws: true,
      }
    }
  }
})
