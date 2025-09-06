"""
Web Dashboard for Solana Trading Bot 2025

Provides a web interface for monitoring bot performance and status.
"""

import asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Dict, Any, List
import json
import uvicorn
from pathlib import Path

# Create FastAPI app
app = FastAPI(title="Solana Trading Bot 2025 Dashboard")

# Setup templates and static files
templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# WebSocket connections
active_connections: List[WebSocket] = []

# Bot status (would be updated by actual bot instances)
bot_status = {
    "sniper_bot": {
        "running": False,
        "status": "stopped",
        "stats": {
            "tokens_scanned": 0,
            "signals_generated": 0,
            "trades_executed": 0,
            "total_profit": 0.0
        }
    },
    "copy_bot": {
        "running": False,
        "status": "stopped",
        "stats": {
            "tracked_traders": 0,
            "copied_trades": 0,
            "success_rate": 0.0
        }
    }
}

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/status")
async def get_status():
    """Get bot status API endpoint"""
    return bot_status

@app.get("/api/performance")
async def get_performance():
    """Get performance metrics API endpoint"""
    # This would fetch actual performance data
    performance_data = {
        "total_profit": 0.0,
        "daily_pnl": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "open_positions": 0
    }
    return performance_data

@app.get("/api/recent_trades")
async def get_recent_trades():
    """Get recent trades API endpoint"""
    # This would fetch actual trade history
    recent_trades = []
    return recent_trades

@app.get("/api/tokens")
async def get_tokens():
    """Get monitored tokens API endpoint"""
    # This would fetch actual token data
    tokens = []
    return tokens

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Send periodic updates
            await websocket.send_text(json.dumps(bot_status))
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def broadcast_update(data: Dict[str, Any]):
    """Broadcast update to all WebSocket connections"""
    message = json.dumps(data)
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except:
            # Remove dead connections
            active_connections.remove(connection)

async def start_dashboard(port: int = 8000):
    """Start the dashboard server"""
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(start_dashboard()) 