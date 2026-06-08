/**
 * SmartShield Extension — Vite Build Configuration
 * Builds all extension entry points into dist/ for Chrome/Firefox.
 */

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig(({ mode }) => ({
  plugins: [react()],

  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: mode === "development",
    minify: mode === "production",

    rollupOptions: {
      input: {
        popup:          resolve(__dirname, "src/popup/index.html"),
        background:     resolve(__dirname, "src/background/service_worker.js"),
        content_script: resolve(__dirname, "src/content/content_script.js"),
        options:        resolve(__dirname, "src/options/index.html"),
      },
      output: {
        entryFileNames: "[name].js",
        chunkFileNames: "[name].js",
        assetFileNames: "[name].[ext]",
      },
    },
  },

  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },

  define: {
    "process.env.NODE_ENV": JSON.stringify(mode),
    "process.env.API_BASE_URL": JSON.stringify(
      mode === "development"
        ? "http://localhost:8000/api/v1"
        : "https://smartshield-api.yourdomain.com/api/v1"
    ),
  },
}));
