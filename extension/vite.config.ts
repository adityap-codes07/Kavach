import webExtension from "vite-plugin-web-extension";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    webExtension({
      manifest: "manifest.json",
    }),
  ],

  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: mode === "development",
    minify: mode === "production",
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
        : "https://kavach-api.yourdomain.com/api/v1"
    ),
  },
}));
