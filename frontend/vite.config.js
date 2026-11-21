import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  // Tailwind is only used by the command-centre dashboard (see src/dashboard).
  plugins: [react(), tailwindcss()],
})
