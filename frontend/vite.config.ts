import { defineConfig, PluginOption } from 'vite'
import react from '@vitejs/plugin-react-swc'
import { visualizer } from 'rollup-plugin-visualizer'

export default defineConfig({
  plugins: [react(), visualizer() as PluginOption],
  build: {
    minify: 'esbuild'
  }
})
