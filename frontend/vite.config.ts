import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/predict': 'http://127.0.0.1:5000',
      '/generate-poster': 'http://127.0.0.1:5000',
      '/api': 'http://127.0.0.1:5000',
      '/fruit-sample': 'http://127.0.0.1:5000',
    },
  },
})
