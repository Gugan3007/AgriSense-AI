import axios from 'axios';

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
});

export const assetUrl = (path) => {
  if (!path) return '';
  if (path.startsWith('http')) return path;
  return `${API_BASE_URL}${path}`;
};

export const stressColors = {
  Healthy: '#6FE27D',
  Low: '#7DDCFF',
  Medium: '#F4B400',
  High: '#E14B4B',
};

export const sensorColumns = ['Soil_Moisture', 'Temperature', 'Humidity', 'Light_Intensity'];
