import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, UTCTimestamp } from 'lightweight-charts';
import { ArrowUpCircle, ArrowDownCircle, Activity } from 'lucide-react';
import useWebSocket, { ReadyState } from 'react-use-websocket';

const SOCKET_URL = 'ws://localhost:8000/ws';

interface ChartData {
    time: UTCTimestamp;
    open: number;
    high: number;
    low: number;
    close: number;
}

interface Signal {
    signal_name: string;
    confidence: number;
    direction: number;
    entry_price: number;
    stop_loss: number;
    take_profit: number;
    timestamp: string;
}

interface Trade {
    symbol: string;
    pnl: number;
    pnl_pct: number;
    exit_reason: string;
    timestamp: string;
}

export const TradingDashboard: React.FC = () => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

    const [data, setData] = useState<ChartData[]>([]);
    const [lastSignal, setLastSignal] = useState<Signal | null>(null);
    const [lastTrade, setLastTrade] = useState<Trade | null>(null);
    const [timeframe, setTimeframe] = useState<string>('5m'); // Timeframe state
    const [equity, setEquity] = useState<number>(100000);
    const [openPositions, setOpenPositions] = useState<number>(0);
    const [isConnected, setIsConnected] = useState(false);
    const [signalMarkers, setSignalMarkers] = useState<any[]>([]);

    const { sendJsonMessage, lastJsonMessage, readyState } = useWebSocket(SOCKET_URL, {
        onOpen: () => setIsConnected(true),
        onClose: () => setIsConnected(false),
        shouldReconnect: () => true,
    });

    // Initialize Chart
    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: '#1a1a1a' },
                textColor: '#d1d5db',
            },
            grid: {
                vertLines: { color: '#2b2b2b' },
                horzLines: { color: '#2b2b2b' },
            },
            width: chartContainerRef.current.clientWidth,
            height: 500,
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            },
        });

        const candleSeries = chart.addCandlestickSeries({
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        });

        chartRef.current = chart;
        candleSeriesRef.current = candleSeries;

        const handleResize = () => {
            chart.applyOptions({ width: chartContainerRef.current?.clientWidth || 800 });
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, []);

    // Handle WebSocket Message
    useEffect(() => {
        if (lastJsonMessage) {
            const msg = lastJsonMessage as any;

            if (msg.type === 'history' && candleSeriesRef.current) {
                const historyData = msg.data.map((item: any) => ({
                    time: item.time as UTCTimestamp,
                    open: item.open,
                    high: item.high,
                    low: item.low,
                    close: item.close
                }));
                candleSeriesRef.current.setData(historyData);
                // Also fit content
                chartRef.current?.timeScale().fitContent();
            }

            if (msg.type === 'market_update' && candleSeriesRef.current) {
                const market = msg.data;
                // Parse timestamp correctly
                const timestamp = new Date(market.timestamp).getTime() / 1000;

                const candle = {
                    time: timestamp as UTCTimestamp,
                    open: market.open,
                    high: market.high,
                    low: market.low,
                    close: market.close,
                };

                // Update Chart
                candleSeriesRef.current.update(candle);

                setEquity(market.equity);
                setOpenPositions(market.open_positions);
            }

            if (msg.type === 'signal') {
                const signal = msg.data;
                setLastSignal(signal);

                // Add marker to chart (APPENDING to existing)
                if (candleSeriesRef.current && signal.signal !== 0) {
                    const newMarker = {
                        time: (new Date(signal.timestamp).getTime() / 1000) as UTCTimestamp,
                        position: signal.signal === 1 ? 'belowBar' : 'aboveBar',
                        color: signal.signal === 1 ? '#2196F3' : '#E91E63',
                        shape: signal.signal === 1 ? 'arrowUp' : 'arrowDown',
                        text: signal.signal_name,
                    };

                    // We need to keep track of markers to append. 
                    // For simplicity, we can just ask the backend for history again, or simpler:
                    // Since we don't have easy access to current markers list here without state,
                    // let's rely on the fact that signal_history sets the initial state.
                    // Ideally we should have a state for markers.
                    // BUT, to keep it robust and simple:
                    // We can just rely on the NEXT update loop or signal_history functionality.
                    // However, to fix the user's immediate request:
                    setSignalMarkers(prev => {
                        // Remove any existing marker for this timestamp to prevent stacking
                        const filtered = prev.filter(m => m.time !== newMarker.time);
                        const updated = [...filtered, newMarker];
                        // Sort by time is required by lightweight-charts
                        updated.sort((a, b) => (a.time as number) - (b.time as number));
                        candleSeriesRef.current?.setMarkers(updated);
                        return updated;
                    });
                }
            }

            if (msg.type === 'signal_history' && candleSeriesRef.current) {
                const signals = msg.data;
                // Filter signals that have .signal property and ensure it's not 0
                const markers = signals
                    .filter((s: any) => s.signal !== undefined && s.signal !== 0)
                    .map((s: any) => ({
                        time: (new Date(s.timestamp).getTime() / 1000) as UTCTimestamp,
                        position: s.signal === 1 ? 'belowBar' : 'aboveBar',
                        color: s.signal === 1 ? '#2196F3' : '#E91E63',
                        shape: s.signal === 1 ? 'arrowUp' : 'arrowDown',
                        text: s.signal_name,
                    }));
                candleSeriesRef.current.setMarkers(markers);
                setSignalMarkers(markers);
            }

            if (msg.type === 'trade') {
                setLastTrade(msg.data);
            }
        }
    }, [lastJsonMessage]);

    const handleStart = async (selectedTimeframe?: string) => {
        // Use provided timeframe or current state
        const tfToUse = selectedTimeframe || timeframe;

        await fetch('http://localhost:8000/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol: '^NSEI', timeframe: tfToUse })
        });
    };

    // Auto-switch when timeframe button is clicked
    const switchTimeframe = (newTimeframe: string) => {
        setTimeframe(newTimeframe);
        handleStart(newTimeframe); // Trigger immediately with new value
    };

    const handleStop = async () => {
        await fetch('http://localhost:8000/stop', { method: 'POST' });
    };

    return (
        <div className="min-h-screen bg-gray-900 text-white p-6 font-sans">
            <header className="flex justify-between items-center mb-6">
                <h1 className="text-2xl font-bold flex items-center gap-2">
                    <Activity className="text-blue-500" />
                    Trading AI Pro
                </h1>

                {/* Timeframe Selector */}
                <div className="flex bg-gray-800 rounded-lg p-1 gap-1">
                    {['1m', '3m', '5m', '10m', '15m', '1h', '1d'].map((tf) => (
                        <button
                            key={tf}
                            onClick={() => switchTimeframe(tf)}
                            className={`px-3 py-1 rounded text-sm font-medium transition-colors ${timeframe === tf
                                ? 'bg-blue-600 text-white'
                                : 'text-gray-400 hover:bg-gray-700 hover:text-white'
                                }`}
                        >
                            {tf}
                        </button>
                    ))}
                </div>

                <div className="flex gap-4 items-center">
                    <span className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></span>
                    <span className="text-gray-400">{isConnected ? 'Connected' : 'Disconnected'}</span>
                </div>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Main Chart Area */}
                <div className="lg:col-span-3 space-y-4">
                    <div
                        ref={chartContainerRef}
                        className="w-full h-[500px] border border-gray-800 rounded-lg overflow-hidden bg-[#1a1a1a]"
                    />

                    <div className="flex justify-end gap-3">
                        <button
                            onClick={() => handleStart()}
                            className="px-6 py-2 bg-green-600 hover:bg-green-700 rounded font-semibold transition-colors"
                        >
                            Start Trading
                        </button>
                        <button
                            onClick={handleStop}
                            className="px-6 py-2 bg-red-600 hover:bg-red-700 rounded font-semibold transition-colors"
                        >
                            Stop Trading
                        </button>
                    </div>
                </div>

                {/* Sidebar Info */}
                <div className="space-y-6">
                    {/* Equity Card */}
                    <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                        <h3 className="text-gray-400 text-sm font-medium mb-1">Total Equity</h3>
                        <div className="text-2xl font-bold text-white">
                            ${equity.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                        </div>
                        <div className={`text-sm mt-1 ${openPositions > 0 ? 'text-blue-400' : 'text-gray-500'}`}>
                            {openPositions} Open Positions
                        </div>
                    </div>

                    {/* Last Signal Card */}
                    <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                        <h3 className="text-gray-400 text-sm font-medium mb-3">Last Signal</h3>
                        {lastSignal ? (
                            <div className="space-y-3">
                                <div className="flex items-center justify-between">
                                    <span className="font-bold text-lg">{lastSignal.signal_name}</span>
                                    {lastSignal.direction === 1 ? (
                                        <ArrowUpCircle className="text-green-500" />
                                    ) : (
                                        <ArrowDownCircle className="text-red-500" />
                                    )}
                                </div>
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                    <div>
                                        <span className="text-gray-500 block">Entry</span>
                                        <span className="font-mono">{lastSignal.entry_price.toFixed(2)}</span>
                                    </div>
                                    <div>
                                        <span className="text-gray-500 block">Conf.</span>
                                        <span className="font-mono">{(lastSignal.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                    <div>
                                        <span className="text-gray-500 block">Target</span>
                                        <span className="font-mono text-green-400">{lastSignal.take_profit.toFixed(2)}</span>
                                    </div>
                                    <div>
                                        <span className="text-gray-500 block">Stop</span>
                                        <span className="font-mono text-red-400">{lastSignal.stop_loss.toFixed(2)}</span>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="text-gray-500 italic text-sm text-center py-4">
                                Waiting for signal...
                            </div>
                        )}
                    </div>

                    {/* Last Trade Card */}
                    <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                        <h3 className="text-gray-400 text-sm font-medium mb-3">Last Trade</h3>
                        {lastTrade ? (
                            <div>
                                <div className="flex justify-between mb-2">
                                    <span className="font-semibold">{lastTrade.symbol}</span>
                                    <span className={`font-bold ${lastTrade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                        {lastTrade.pnl >= 0 ? '+' : ''}{lastTrade.pnl.toFixed(2)}
                                    </span>
                                </div>
                                <div className="text-xs text-gray-500">
                                    Exit: {lastTrade.exit_reason}
                                </div>
                            </div>
                        ) : (
                            <div className="text-gray-500 italic text-sm text-center py-4">
                                No trades yet
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default TradingDashboard;
