export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        forest: '#0F3D2E',
        pine: '#0A2A1F',
        night: '#041611',
        mist: '#F7F9F4',
        electric: '#2E86FF',
        amber: '#F4B400',
        danger: '#E14B4B',
      },
      fontFamily: {
        display: ['Sora', 'Space Grotesk', 'Inter', 'system-ui', 'sans-serif'],
        body: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        glass: '0 24px 80px rgba(0, 0, 0, 0.35)',
        glow: '0 0 36px rgba(46, 134, 255, 0.28)',
      },
    },
  },
  plugins: [],
};
