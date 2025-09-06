"""
Web Dashboard for AI Market Maker & Liquidation Monitor
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import plotly.graph_objects as go
import plotly.express as px
import plotly.utils
import pandas as pd
import numpy as np

from ..config.settings import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ai-market-maker-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# Global variables for dashboard data
dashboard_data = {
    "positions": [],
    "liquidation_signals": [],
    "market_maker_activities": [],
    "risk_metrics": {},
    "trading_signals": [],
    "performance_metrics": {},
    "system_status": {}
}

# Dashboard update task
dashboard_task = None


class DashboardManager:
    """Manages dashboard data and updates"""
    
    def __init__(self):
        self.is_running = False
        self.update_interval = 5  # seconds
        self.data_sources = {}
        
    async def initialize(self):
        """Initialize dashboard manager"""
        try:
            logger.info("Initializing Dashboard Manager...")
            self.is_running = True
            
            # Start dashboard update loop
            asyncio.create_task(self._dashboard_update_loop())
            
            logger.info("Dashboard Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Dashboard Manager: {e}")
            return False
    
    async def _dashboard_update_loop(self):
        """Main dashboard update loop"""
        while self.is_running:
            try:
                # Update dashboard data
                await self._update_dashboard_data()
                
                # Emit updates to connected clients
                await self._emit_dashboard_updates()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(10)
    
    async def _update_dashboard_data(self):
        """Update dashboard data from various sources"""
        try:
            # Update positions
            dashboard_data["positions"] = await self._get_position_data()
            
            # Update liquidation signals
            dashboard_data["liquidation_signals"] = await self._get_liquidation_signals()
            
            # Update market maker activities
            dashboard_data["market_maker_activities"] = await self._get_market_maker_data()
            
            # Update risk metrics
            dashboard_data["risk_metrics"] = await self._get_risk_metrics()
            
            # Update trading signals
            dashboard_data["trading_signals"] = await self._get_trading_signals()
            
            # Update performance metrics
            dashboard_data["performance_metrics"] = await self._get_performance_metrics()
            
            # Update system status
            dashboard_data["system_status"] = await self._get_system_status()
            
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
    
    async def _get_position_data(self) -> List[Dict[str, Any]]:
        """Get position data"""
        try:
            # This would integrate with position monitor
            # For now, return mock data
            return [
                {
                    "symbol": "BTC",
                    "side": "long",
                    "size": 1.5,
                    "entry_price": 45000,
                    "current_price": 46000,
                    "unrealized_pnl": 1500,
                    "leverage": 2.0,
                    "liquidation_price": 42000,
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "symbol": "ETH",
                    "side": "short",
                    "size": 10.0,
                    "entry_price": 3000,
                    "current_price": 3100,
                    "unrealized_pnl": -1000,
                    "leverage": 1.5,
                    "liquidation_price": 3300,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        except Exception as e:
            logger.error(f"Error getting position data: {e}")
            return []
    
    async def _get_liquidation_signals(self) -> List[Dict[str, Any]]:
        """Get liquidation signals"""
        try:
            # This would integrate with liquidation predictor
            # For now, return mock data
            return [
                {
                    "symbol": "SOL",
                    "side": "long",
                    "probability": 0.85,
                    "confidence": 0.78,
                    "estimated_time": (datetime.now() + timedelta(minutes=30)).isoformat(),
                    "risk_level": "high",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        except Exception as e:
            logger.error(f"Error getting liquidation signals: {e}")
            return []
    
    async def _get_market_maker_data(self) -> List[Dict[str, Any]]:
        """Get market maker activities"""
        try:
            # This would integrate with market maker tracker
            # For now, return mock data
            return [
                {
                    "market_maker_id": "mm_001",
                    "symbol": "BTC",
                    "activity_type": "liquidity_provision",
                    "volume": 500000,
                    "price_impact": 0.001,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        except Exception as e:
            logger.error(f"Error getting market maker data: {e}")
            return []
    
    async def _get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics"""
        try:
            # This would integrate with risk manager
            # For now, return mock data
            return {
                "var_95": -0.025,
                "var_99": -0.035,
                "max_drawdown": -0.15,
                "sharpe_ratio": 1.2,
                "volatility": 0.25,
                "beta": 1.1,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {}
    
    async def _get_trading_signals(self) -> List[Dict[str, Any]]:
        """Get trading signals"""
        try:
            # This would integrate with signal generator
            # For now, return mock data
            return [
                {
                    "symbol": "AVAX",
                    "signal_type": "buy",
                    "confidence": 0.75,
                    "strength": 0.8,
                    "entry_price": 25.0,
                    "target_price": 27.5,
                    "stop_loss": 23.5,
                    "timeframe": "medium",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        except Exception as e:
            logger.error(f"Error getting trading signals: {e}")
            return []
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            # This would calculate actual performance
            # For now, return mock data
            return {
                "total_trades": 150,
                "win_rate": 0.65,
                "profit_factor": 1.8,
                "total_pnl": 25000,
                "max_drawdown": -8000,
                "sharpe_ratio": 1.4,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            return {
                "status": "running",
                "uptime": "2 days, 5 hours",
                "active_components": 6,
                "last_update": datetime.now().isoformat(),
                "errors": 0,
                "warnings": 2
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}
    
    async def _emit_dashboard_updates(self):
        """Emit dashboard updates to connected clients"""
        try:
            # Emit data updates
            socketio.emit('dashboard_update', dashboard_data)
            
            # Emit specific updates
            socketio.emit('positions_update', dashboard_data["positions"])
            socketio.emit('liquidation_signals_update', dashboard_data["liquidation_signals"])
            socketio.emit('risk_metrics_update', dashboard_data["risk_metrics"])
            socketio.emit('trading_signals_update', dashboard_data["trading_signals"])
            
        except Exception as e:
            logger.error(f"Error emitting dashboard updates: {e}")


# Initialize dashboard manager
dashboard_manager = DashboardManager()


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/positions')
def get_positions():
    """Get position data API endpoint"""
    return jsonify(dashboard_data["positions"])


@app.route('/api/liquidation-signals')
def get_liquidation_signals():
    """Get liquidation signals API endpoint"""
    return jsonify(dashboard_data["liquidation_signals"])


@app.route('/api/market-maker-activities')
def get_market_maker_activities():
    """Get market maker activities API endpoint"""
    return jsonify(dashboard_data["market_maker_activities"])


@app.route('/api/risk-metrics')
def get_risk_metrics():
    """Get risk metrics API endpoint"""
    return jsonify(dashboard_data["risk_metrics"])


@app.route('/api/trading-signals')
def get_trading_signals():
    """Get trading signals API endpoint"""
    return jsonify(dashboard_data["trading_signals"])


@app.route('/api/performance-metrics')
def get_performance_metrics():
    """Get performance metrics API endpoint"""
    return jsonify(dashboard_data["performance_metrics"])


@app.route('/api/system-status')
def get_system_status():
    """Get system status API endpoint"""
    return jsonify(dashboard_data["system_status"])


@app.route('/api/charts/price-chart')
def get_price_chart():
    """Get price chart data"""
    try:
        # Generate mock price data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='H')
        prices = 45000 + np.cumsum(np.random.normal(0, 100, len(dates)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='BTC Price',
            line=dict(color='#00ff88', width=2)
        ))
        
        fig.update_layout(
            title='BTC Price Chart (30 Days)',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            height=400
        )
        
        return jsonify(json.loads(fig.to_json()))
        
    except Exception as e:
        logger.error(f"Error generating price chart: {e}")
        return jsonify({"error": str(e)})


@app.route('/api/charts/volume-chart')
def get_volume_chart():
    """Get volume chart data"""
    try:
        # Generate mock volume data
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
        volumes = np.random.lognormal(10, 0.5, len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dates,
            y=volumes,
            name='Trading Volume',
            marker_color='#ff8800'
        ))
        
        fig.update_layout(
            title='Trading Volume (7 Days)',
            xaxis_title='Date',
            yaxis_title='Volume (USD)',
            template='plotly_dark',
            height=300
        )
        
        return jsonify(json.loads(fig.to_json()))
        
    except Exception as e:
        logger.error(f"Error generating volume chart: {e}")
        return jsonify({"error": str(e)})


@app.route('/api/charts/risk-chart')
def get_risk_chart():
    """Get risk chart data"""
    try:
        # Generate mock risk data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        var_95 = -0.02 + np.random.normal(0, 0.005, len(dates))
        var_99 = -0.03 + np.random.normal(0, 0.008, len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=var_95,
            mode='lines',
            name='VaR 95%',
            line=dict(color='#ffaa00', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=dates,
            y=var_99,
            mode='lines',
            name='VaR 99%',
            line=dict(color='#ff4400', width=2)
        ))
        
        fig.update_layout(
            title='Value at Risk (30 Days)',
            xaxis_title='Date',
            yaxis_title='VaR',
            template='plotly_dark',
            height=300
        )
        
        return jsonify(json.loads(fig.to_json()))
        
    except Exception as e:
        logger.error(f"Error generating risk chart: {e}")
        return jsonify({"error": str(e)})


@app.route('/api/charts/performance-chart')
def get_performance_chart():
    """Get performance chart data"""
    try:
        # Generate mock performance data
        dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='D')
        cumulative_pnl = np.cumsum(np.random.normal(100, 50, len(dates)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_pnl,
            mode='lines',
            name='Cumulative PnL',
            line=dict(color='#00ff88', width=2),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title='Cumulative PnL (90 Days)',
            xaxis_title='Date',
            yaxis_title='PnL (USD)',
            template='plotly_dark',
            height=300
        )
        
        return jsonify(json.loads(fig.to_json()))
        
    except Exception as e:
        logger.error(f"Error generating performance chart: {e}")
        return jsonify({"error": str(e)})


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('request_data')
def handle_data_request(data):
    """Handle data requests from clients"""
    try:
        data_type = data.get('type')
        if data_type == 'positions':
            emit('positions_data', dashboard_data["positions"])
        elif data_type == 'liquidation_signals':
            emit('liquidation_signals_data', dashboard_data["liquidation_signals"])
        elif data_type == 'risk_metrics':
            emit('risk_metrics_data', dashboard_data["risk_metrics"])
        elif data_type == 'trading_signals':
            emit('trading_signals_data', dashboard_data["trading_signals"])
        else:
            emit('error', {'message': f'Unknown data type: {data_type}'})
            
    except Exception as e:
        logger.error(f"Error handling data request: {e}")
        emit('error', {'message': str(e)})


async def start_dashboard():
    """Start the dashboard"""
    try:
        # Initialize dashboard manager
        await dashboard_manager.initialize()
        
        # Get settings
        settings = get_settings()
        
        # Start Flask app
        socketio.run(
            app,
            host=settings.web.host,
            port=settings.web.port,
            debug=settings.web.debug,
            use_reloader=False
        )
        
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")


if __name__ == '__main__':
    asyncio.run(start_dashboard()) 