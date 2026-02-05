import asyncio
import json
import logging
import threading
import sys
import os

# Add project root to sys.path to allow imports from siblings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Optional
from datetime import datetime
import pandas as pd

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from live.live_trader import LiveTrader, PaperTrade
from models.trainer import ModelTrainer
from features.feature_pipeline import FeatureEngineer
from risk.risk_manager import RiskManager
from config import MODELS_DIR, DEFAULT_RISK_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="Trading AI API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TradeManager:
    """Manages the LiveTrader instance and WebSocket connections."""
    
    def __init__(self):
        self.trader: Optional[LiveTrader] = None
        self.active_connections: list[WebSocket] = []
        self.latest_state: Dict = {}
        self.history: list = []  # Cache for historical data
        self.is_running = False
        self._thread: Optional[threading.Thread] = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Send history first if available
        if self.history:
            print(f"DEBUG: Sending complete history ({len(self.history)} records) to new client")
            await websocket.send_json({'type': 'history', 'data': self.history})
            
        # Send initial state if available
        if self.latest_state:
            await websocket.send_json(self.latest_state)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")

    def on_update(self, data: Dict):
        """Callback for LiveTrader updates."""
        self.latest_state['market'] = data
        asyncio.run_coroutine_threadsafe(
            self.broadcast({'type': 'market_update', 'data': data}),
            loop
        )

    def on_history(self, df):
        """Callback for broadcasting historical data."""
        print(f"DEBUG: on_history called with {len(df)} rows")
        try:
            # Convert DataFrame to list of dicts for the chart
            records = []
            for index, row in df.iterrows():
                # Handle index as timestamp if needed, or column
                ts = index if isinstance(index, datetime) else pd.to_datetime(row['timestamp'])
                records.append({
                    'time': ts.timestamp(), # Unix timestamp for lightweight-charts
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                })
            
            # Cache history
            self.history = records
            
            print(f"DEBUG: Broadcasting {len(records)} history records")
            asyncio.run_coroutine_threadsafe(
                self.broadcast({'type': 'history', 'data': records}),
                loop
            )
        except Exception as e:
            print(f"DEBUG: Error in on_history: {e}")
            import traceback
            traceback.print_exc()

    def on_signal(self, signal):
        """Callback for new signals."""
        data = {
            'signal_name': signal.signal_name,
            'confidence': signal.confidence,
            'direction': signal.signal,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'timestamp': signal.timestamp.isoformat() if hasattr(signal.timestamp, 'isoformat') else str(signal.timestamp)
        }
        self.latest_state['last_signal'] = data
        asyncio.run_coroutine_threadsafe(
            self.broadcast({'type': 'signal', 'data': data}),
            loop
        )

    def on_trade(self, trade: PaperTrade):
        """Callback for completed trades."""
        data = {
            'symbol': trade.symbol,
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct,
            'exit_reason': trade.exit_reason,
            'timestamp': datetime.now().isoformat()
        }
        asyncio.run_coroutine_threadsafe(
            self.broadcast({'type': 'trade', 'data': data}),
            loop
        )

    def on_signal_history(self, history: list):
        """Callback for signal history."""
        asyncio.run_coroutine_threadsafe(
            self.broadcast({'type': 'signal_history', 'data': history}),
            loop
        )

    def start_trader(self, symbol: str, timeframe: str):
        """Start the LiveTrader in a background thread."""
        if self.is_running:
            return

        # Load resources
        model_path = MODELS_DIR / f"{symbol}_xgboost_model.joblib"
        pipeline_path = MODELS_DIR / f"{symbol}_pipeline"

        if not model_path.exists():
            raise ValueError(f"Model not found for {symbol}")

        trainer = ModelTrainer()
        trainer.load(model_path)

        engineer = FeatureEngineer()
        engineer.load(pipeline_path)

        risk_manager = RiskManager(
            initial_capital=100000.0,
            max_risk_per_trade=DEFAULT_RISK_CONFIG.max_risk_per_trade,
            max_daily_loss=DEFAULT_RISK_CONFIG.max_daily_loss,
            max_drawdown=DEFAULT_RISK_CONFIG.max_drawdown
        )

        self.trader = LiveTrader(
            model_trainer=trainer,
            feature_engineer=engineer,
            risk_manager=risk_manager,
            initial_capital=100000.0,
            symbol=symbol,
            data_source='yahoo',
            timeframe=timeframe,
            update_interval=5 # Faster updates for UI demo
        )

        # Hook up callbacks
        self.trader.on_update = self.on_update
        self.trader.on_signal = self.on_signal
        self.trader.on_trade = self.on_trade
        self.trader.on_history = self.on_history
        self.trader.on_signal_history = self.on_signal_history

        self.is_running = True
        self._thread = threading.Thread(target=self.trader.start, daemon=True)
        self._thread.start()
        logger.info(f"Started trader for {symbol} ({timeframe})")

    def stop_trader(self):
        if self.trader:
            self.trader.stop()
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)

manager = TradeManager()
loop = asyncio.new_event_loop() # Will be set on startup

@app.on_event("startup")
def startup_event():
    global loop
    loop = asyncio.get_running_loop()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep connection open
    except WebSocketDisconnect:
        manager.disconnect(websocket)

from pydantic import BaseModel

class StartRequest(BaseModel):
    symbol: str = "^NSEI"
    timeframe: str = "5m"

@app.post("/start")
async def start_trading(request: StartRequest):
    try:
        print(f"\n[USER VERIFICATION] Switching Timeframe to: {request.timeframe.upper()}")
        print(f"[USER VERIFICATION] Restarting Trading Engine...")
        manager.start_trader(request.symbol, request.timeframe)
        return {"status": "started", "symbol": request.symbol}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/stop")
async def stop_trading():
    manager.stop_trader()
    return {"status": "stopped"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
