import "tailwindcss";

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        trading: {
          bg: '#0a0f1c',
          card: '#1a2332',
          border: '#2d3748',
          accent: '#4fd1c7',
          green: '#10b981',
          red: '#ef4444',
          yellow: '#f59e0b',
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s ease-in-out infinite',
      }
    },
  },
  plugins: [],
}