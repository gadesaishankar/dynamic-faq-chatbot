import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../web',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/chat': 'http://127.0.0.1:8000',
      '/faq': 'http://127.0.0.1:8000',
      '/feedback': 'http://127.0.0.1:8000',
      '/analytics': 'http://127.0.0.1:8000',
      '/categories': 'http://127.0.0.1:8000',
      '/admin': 'http://127.0.0.1:8000',
      '/ingest': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
})
