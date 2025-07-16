// src/App.jsx
import { useState, useEffect } from 'react';
import { nanoid } from 'nanoid'; // Импортируем nanoid для уникальных ключей
import './App.css';
import PriceChart from './PriceChart'; // Импортируем наш новый компонент графика

function App() {
  const [balance, setBalance] = useState(10000);
  const [pnl, setPnl] = useState(0);
  const [currentPrice, setCurrentPrice] = useState(2350.50);
  const [priceHistory, setPriceHistory] = useState(() => 
    Array.from({ length: 50 }, (_, i) => 2350.50 + (Math.random() - 0.5) * (i / 5))
  );
  const [botStatus, setBotStatus] = useState('active');
  const [logs, setLogs] = useState([
    { id: nanoid(), time: '14:32:15', message: '🤖 AI Model loaded successfully', type: 'success' },
    { id: nanoid(), time: '14:32:20', message: '📊 Market data streaming...', type: 'info' },
    { id: nanoid(), time: '14:32:25', message: '🎯 BUY signal detected - Confidence: 87%', type: 'decision' },
    { id: nanoid(), time: '14:32:30', message: '✅ Position opened: GOLD @ $2350.50', type: 'trade' },
  ]);
  
  // Состояние для открытой позиции
  const [openPosition, setOpenPosition] = useState({
    symbol: 'GOLD',
    units: 1.0,
    entryPrice: 2350.50,
  });

  // Состояние для сигналов ИИ
  const [aiSignal, setAiSignal] = useState({
      rsi: 45.2,
      sentiment: 0.7,
      volatility: 'High',
      action: 'HOLD',
  });

  // Симуляция обновления цены и данных
  useEffect(() => {
    const interval = setInterval(() => {
      let newPrice = 0;
      setCurrentPrice(prev => {
        newPrice = prev + (Math.random() - 0.5) * 2;
        setPriceHistory(history => [...history.slice(1), newPrice]);
        return newPrice;
      });

      setPnl(prev => prev + (Math.random() - 0.5) * 10);

      // Симуляция изменения сигналов ИИ
      setAiSignal({
          rsi: (40 + Math.random() * 20).toFixed(1),
          sentiment: (Math.random() * 2 - 1).toFixed(2),
          volatility: Math.random() > 0.5 ? 'High' : 'Medium',
          action: Math.random() > 0.9 ? 'BUY' : (Math.random() < 0.1 ? 'SELL' : 'HOLD'),
      });

    }, 2000);
    
    return () => clearInterval(interval);
  }, []);

  const addLog = (message, type = 'info') => {
    const newLog = {
      id: nanoid(),
      time: new Date().toLocaleTimeString(),
      message,
      type
    };
    setLogs(prev => [newLog, ...prev].slice(0, 10));
  };
  
  const positionPnl = openPosition ? (currentPrice - openPosition.entryPrice) * openPosition.units : 0;
  const positionPnlPercent = openPosition ? (positionPnl / (openPosition.entryPrice * openPosition.units)) * 100 : 0;


  return (
    <div className="min-h-screen bg-trading-bg text-white p-4 font-sans">
      {/* Header */}
      <header className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white">
              🤖 <span className="text-trading-accent">AI Trading Bot</span>
            </h1>
            <p className="text-gray-400">Real-time algorithmic trading dashboard</p>
          </div>
          <div className="flex items-center space-x-4">
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${
              botStatus === 'active' 
                ? 'bg-trading-green/20 text-trading-green' 
                : 'bg-trading-red/20 text-trading-red'
            }`}>
              <span className={`inline-block w-2 h-2 rounded-full mr-2 ${
                botStatus === 'active' ? 'bg-trading-green' : 'bg-trading-red'
              } animate-pulse`}></span>
              {botStatus === 'active' ? 'Bot Active' : 'Bot Offline'}
            </div>
          </div>
        </div>
      </header>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-trading-card/50 backdrop-blur-md border border-trading-border rounded-xl p-6 hover:border-trading-accent/50 transition-all">
          <h3 className="text-gray-400 text-sm mb-2">Portfolio Balance</h3>
          <p className="text-2xl font-bold text-white">${balance.toLocaleString()}</p>
          <p className="text-xs text-gray-500 mt-1">Available: ${(balance * 0.95).toLocaleString()}</p>
        </div>
        
        <div className="bg-trading-card/50 backdrop-blur-md border border-trading-border rounded-xl p-6 hover:border-trading-accent/50 transition-all">
          <h3 className="text-gray-400 text-sm mb-2">P&L Today</h3>
          <p className={`text-2xl font-bold ${pnl >= 0 ? 'text-trading-green' : 'text-trading-red'}`}>
            ${pnl.toFixed(2)}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            {((pnl / balance) * 100).toFixed(2)}%
          </p>
        </div>
        
        <div className="bg-trading-card/50 backdrop-blur-md border border-trading-border rounded-xl p-6 hover:border-trading-accent/50 transition-all">
          <h3 className="text-gray-400 text-sm mb-2">GOLD Price</h3>
          <p className="text-2xl font-bold text-trading-accent">${currentPrice.toFixed(2)}</p>
          <p className="text-xs text-trading-green mt-1">+0.34% (24h)</p>
        </div>
        
        <div className="bg-trading-card/50 backdrop-blur-md border border-trading-border rounded-xl p-6 hover:border-trading-accent/50 transition-all">
          <h3 className="text-gray-400 text-sm mb-2">AI Confidence</h3>
          <p className="text-2xl font-bold text-trading-yellow">87%</p>
          <p className="text-xs text-gray-500 mt-1">High confidence signal</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Control Panel (AI Engine & Positions) */}
        <div className="lg:col-span-1 bg-trading-card/50 backdrop-blur-md border border-trading-border rounded-xl p-6">
            {/* AI Decision Engine */}
            <div>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">🧠 AI Decision Engine</h3>
                  <div className="w-2 h-2 bg-trading-accent rounded-full animate-pulse"></div>
                </div>
                
                <div className="space-y-4">
                  <div className="flex justify-between items-center py-2 border-b border-trading-border/50">
                    <span className="text-gray-400">Current Action</span>
                    <span className={`font-medium ${aiSignal.action === 'BUY' ? 'text-trading-green' : aiSignal.action === 'SELL' ? 'text-trading-red' : 'text-trading-yellow'}`}>{aiSignal.action}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-trading-border/50">
                    <span className="text-gray-400">RSI Signal</span>
                    <span className="text-trading-yellow">Neutral ({aiSignal.rsi})</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-trading-border/50">
                    <span className="text-gray-400">Sentiment</span>
                    <span className={aiSignal.sentiment > 0 ? 'text-trading-green' : 'text-trading-red'}>{aiSignal.sentiment > 0 ? 'Bullish' : 'Bearish'} ({aiSignal.sentiment})</span>
                  </div>
                  <div className="flex justify-between items-center py-2">
                    <span className="text-gray-400">Market Volatility</span>
                    <span className={aiSignal.volatility === 'High' ? 'text-trading-red' : 'text-trading-yellow'}>{aiSignal.volatility}</span>
                  </div>
                </div>

                <button 
                  onClick={() => addLog('🔄 Manual analysis triggered', 'info')}
                  className="w-full mt-6 bg-trading-accent hover:bg-trading-accent/80 text-trading-bg font-medium py-2 px-4 rounded-lg transition-colors"
                >
                  Trigger Analysis
                </button>
            </div>

            <hr className="my-6 border-trading-border/50" />

            {/* Open Positions */}
            <div>
                <h3 className="text-lg font-semibold text-white mb-4">📈 Open Positions</h3>
                {openPosition ? (
                    <div className="space-y-3">
                      <div className="flex justify-between items-center p-3 bg-trading-bg/50 rounded-lg">
                        <div>
                          <p className="font-medium text-white">{openPosition.symbol}</p>
                          <p className="text-xs text-gray-400">{openPosition.units} units @ ${openPosition.entryPrice.toFixed(2)}</p>
                        </div>
                        <div className="text-right">
                          <p className={`font-medium ${positionPnl >= 0 ? 'text-trading-green' : 'text-trading-red'}`}>
                            {positionPnl >= 0 ? '+' : ''}${positionPnl.toFixed(2)}
                          </p>
                          <p className={`text-xs ${positionPnl >= 0 ? 'text-trading-green' : 'text-trading-red'}`}>
                            {positionPnlPercent.toFixed(2)}%
                          </p>
                        </div>
                      </div>
                    </div>
                ) : (
                    <div className="text-center py-4 text-gray-500 text-sm">
                        No open positions
                    </div>
                )}
            </div>
        </div>

        {/* Chart Area & Logs */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-trading-card/50 backdrop-blur-md border border-trading-border rounded-xl p-6">
            <h3 className="text-lg font-semibold text-white mb-4">📊 Price Chart (GOLD/USD)</h3>
            <div className="h-64 bg-trading-bg/50 rounded-lg p-2">
              <PriceChart data={priceHistory} />
            </div>
          </div>

          <div className="bg-trading-card/50 backdrop-blur-md border border-trading-border rounded-xl p-6">
            <h3 className="text-lg font-semibold text-white mb-4">📝 Live Activity Log</h3>
            <div className="h-48 overflow-y-auto space-y-2 pr-2">
              {logs.map((log) => (
                <div key={log.id} className={`p-2 rounded text-sm border-l-2 ${
                  log.type === 'success' ? 'border-trading-green text-trading-green bg-trading-green/10' :
                  log.type === 'info' ? 'border-blue-400 text-blue-300 bg-blue-400/10' :
                  log.type === 'decision' ? 'border-trading-yellow text-trading-yellow bg-trading-yellow/10' :
                  log.type === 'trade' ? 'border-trading-accent text-trading-accent bg-trading-accent/10' :
                  'border-gray-400 text-gray-300 bg-gray-400/10'
                }`}>
                  <span className="text-gray-400 text-xs mr-2">{log.time}</span>
                  {log.message}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
