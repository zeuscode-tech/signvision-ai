import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

declare const process: { env: Record<string, string | undefined> }

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: Number(process.env.SIGNVISION_FRONTEND_PORT || 5173),
    proxy: {
      '/api': process.env.SIGNVISION_BACKEND_ORIGIN || 'http://127.0.0.1:8000',
      '/ws': { target: process.env.SIGNVISION_BACKEND_ORIGIN || 'http://127.0.0.1:8000', ws: true },
    },
  },
})

