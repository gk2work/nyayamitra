import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // NyayaMitra brand colors
        navy: {
          50: "#f0f4f8",
          100: "#d9e2ec",
          200: "#bcccdc",
          300: "#9fb3c8",
          400: "#829ab1",
          500: "#627d98",
          600: "#486581",
          700: "#334e68",
          800: "#243b53",
          900: "#1b3a5c",
          950: "#102a43",
        },
        saffron: {
          50: "#fff8f0",
          100: "#feecdc",
          200: "#fcd9b8",
          300: "#f9b882",
          400: "#f5924e",
          500: "#d4740a",
          600: "#b45e08",
          700: "#8c4906",
          800: "#6d3904",
          900: "#4e2903",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;