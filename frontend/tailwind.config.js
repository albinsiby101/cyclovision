/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Stitch "Deep Space Atmospheric Intelligence" palette
        "page": "#08090A",
        "surface": "#0E1013",
        "surface2": "#14171C",
        "canvas": "#0E1013",
        "canvas-dim": "#14171C",
        "hairline": "#1E2228",
        "ink": "#FFFFFF",
        "ink-soft": "#B4B7BD",
        "ink-faint": "#8E929B",
        "ink-ghost": "#474B53",
        "apple": "#38BDF8",
        "apple-hover": "#0EA5E9",
        "apple-blue": "#38BDF8",
        "apple-red": "#EF4444",
        "apple-red-soft": "rgba(239,68,68,0.06)",
        "apple-amber": "#F59E0B",
        "apple-amber-soft": "rgba(245,158,11,0.06)",
        "apple-green": "#10B981",
        "apple-green-soft": "rgba(16,185,129,0.06)",
        "violet": "#A855F7"
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
        display: ["'Hanken Grotesk'", "Inter", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["'JetBrains Mono'", "SF Mono", "ui-monospace", "Consolas", "monospace"]
      },
      boxShadow: {
        "card": "0 1px 2px rgba(0,0,0,0.5)",
        "lift": "0 16px 32px -8px rgba(0,0,0,0.7)",
        "edge-white": "0 0 12px rgba(255,255,255,0.2)"
      },
      borderRadius: {
        "2.5xl": "1.25rem"
      }
    },
  },
  plugins: [],
}
