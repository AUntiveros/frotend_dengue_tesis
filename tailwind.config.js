/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        risk: {
          none:  '#1e40af',
          low:   '#0ea5e9',
          mid:   '#eab308',
          high:  '#f97316',
          vhigh: '#dc2626',
        }
      }
    }
  },
  plugins: [],
}
