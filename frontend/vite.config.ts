import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import { visualizer } from 'rollup-plugin-visualizer' // ✅ esta línea te falta

export default defineConfig({
  plugins: [react(), visualizer()],
  build: {
    minify: 'esbuild', // o 'terser'
  }
})
