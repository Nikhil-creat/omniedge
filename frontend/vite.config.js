import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages serves project sites from https://<user>.github.io/<repo>/,
// so every asset URL must be prefixed with `/<repo>/`. Set OMNIEDGE_REPO_NAME
// at build time (the deploy workflow does this automatically from the repo
// name), or hardcode it below if you prefer.
const repoName = process.env.OMNIEDGE_REPO_NAME || "omniedge";

export default defineConfig({
  plugins: [react()],
  base: `/${repoName}/`,
  build: {
    outDir: "dist",
    sourcemap: true,
  },
  server: {
    port: 5173,
  },
});
