/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        void: "#0A0E14",
        panel: "#10151F",
        panelline: "#1E2634",
        signal: "#38F5C8",
        pulse: "#7C6CF6",
        warn: "#F5A623",
        danger: "#F5456C",
        mist: "#8994A8",
      },
      fontFamily: {
        display: ["'Space Grotesk'", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        glow: "0 0 24px -4px rgba(56, 245, 200, 0.35)",
      },
    },
  },
  plugins: [],
};
