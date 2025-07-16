// src/PriceChart.jsx
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

// Нам нужно зарегистрировать компоненты, которые мы собираемся использовать
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const PriceChart = ({ data: priceHistory }) => {
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: false,
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: '#1a2332',
        titleColor: '#a0aec0',
        bodyColor: '#cbd5e0',
        borderColor: '#2d3748',
        borderWidth: 1,
      }
    },
    scales: {
      x: {
        display: false, // Скрываем метки по оси X для более чистого вида
        grid: {
          display: false,
        },
      },
      y: {
        position: 'right', // Перемещаем ось Y вправо
        grid: {
          color: 'rgba(45, 55, 72, 0.5)',
          borderColor: 'transparent',
        },
        ticks: {
          color: '#a0aec0',
          // Форматируем метки как валюту
          callback: function(value) {
            return '$' + value.toFixed(2);
          }
        },
      },
    },
    elements: {
      line: {
        tension: 0.4, // Сглаживаем линию
      },
      point: {
        radius: 0, // Скрываем точки на линии
      }
    },
  };

  const data = {
    // Создаем метки для каждой точки данных, чтобы всплывающая подсказка работала корректно
    labels: priceHistory.map((_, index) => `Point ${index + 1}`),
    datasets: [
      {
        label: 'Price',
        data: priceHistory,
        fill: true,
        // Создаем градиентный фон
        backgroundColor: (context) => {
          if (!context.chart.chartArea) {
            return;
          }
          const ctx = context.chart.ctx;
          const gradient = ctx.createLinearGradient(0, context.chart.chartArea.top, 0, context.chart.chartArea.bottom);
          gradient.addColorStop(0, 'rgba(79, 209, 199, 0.3)');
          gradient.addColorStop(1, 'rgba(79, 209, 199, 0)');
          return gradient;
        },
        borderColor: '#4fd1c7',
        borderWidth: 2,
      },
    ],
  };

  return <Line options={options} data={data} />;
};

export default PriceChart;
