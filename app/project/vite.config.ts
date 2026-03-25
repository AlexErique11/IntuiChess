import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  // Use relative paths in production so Electron's file:// can load assets
  base: mode === 'production' ? './' : '/',
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
}));
