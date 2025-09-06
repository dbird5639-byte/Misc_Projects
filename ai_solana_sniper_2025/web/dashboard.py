"""
Web Dashboard for AI-Powered Solana Meme Coin Sniper
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO, emit
import threading

from ..utils.logger import get_logger
from ..config.settings import get_settings

logger = get_logger(__name__)


class Dashboard:
    """
    Web dashboard for monitoring AI sniper system
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.host = config.get("host", "0.0.0.0")
        self.port = config.get("port", 5000)
        self.debug = config.get("debug", False)
        
        # Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = config.get("secret_key", "ai-sniper-secret-key")
        
        # SocketIO for real-time updates
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # System components (will be set by main application)
        self.sniper_agent = None
        self.chat_agent = None
        self.focus_agent = None
        self.risk_manager = None
        self.market_data = None
        
        # Dashboard state
        self.is_running = False
        self.last_update = datetime.now()
        self.system_status = "stopped"
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio_events()
        
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def api_status():
            """Get system status"""
            return jsonify(self._get_system_status())
        
        @self.app.route('/api/performance')
        def api_performance():
            """Get performance metrics"""
            return jsonify(self._get_performance_metrics())
        
        @self.app.route('/api/positions')
        def api_positions():
            """Get current positions"""
            return jsonify(self._get_positions())
        
        @self.app.route('/api/orders')
        def api_orders():
            """Get order history"""
            return jsonify(self._get_orders())
        
        @self.app.route('/api/risk')
        def api_risk():
            """Get risk metrics"""
            return jsonify(self._get_risk_metrics())
        
        @self.app.route('/api/market')
        def api_market():
            """Get market data"""
            return jsonify(self._get_market_data())
        
        @self.app.route('/api/ai')
        def api_ai():
            """Get AI model status"""
            return jsonify(self._get_ai_status())
        
        @self.app.route('/api/chat')
        def api_chat():
            """Get chat history"""
            return jsonify(self._get_chat_history())
        
        @self.app.route('/api/chat/send', methods=['POST'])
        def api_chat_send():
            """Send chat message"""
            data = request.get_json()
            message = data.get('message', '')
            
            if message and self.chat_agent:
                # Send message asynchronously
                asyncio.create_task(self._send_chat_message(message))
                return jsonify({"status": "sent"})
            
            return jsonify({"status": "error", "message": "Invalid message"})
        
        @self.app.route('/api/control/start', methods=['POST'])
        def api_control_start():
            """Start the system"""
            if self.sniper_agent and not self.sniper_agent.is_running:
                asyncio.create_task(self.sniper_agent.start())
                return jsonify({"status": "started"})
            return jsonify({"status": "error", "message": "Already running or not available"})
        
        @self.app.route('/api/control/stop', methods=['POST'])
        def api_control_stop():
            """Stop the system"""
            if self.sniper_agent and self.sniper_agent.is_running:
                asyncio.create_task(self.sniper_agent.stop())
                return jsonify({"status": "stopped"})
            return jsonify({"status": "error", "message": "Not running or not available"})
        
        @self.app.route('/api/control/restart', methods=['POST'])
        def api_control_restart():
            """Restart the system"""
            if self.sniper_agent:
                asyncio.create_task(self._restart_system())
                return jsonify({"status": "restarting"})
            return jsonify({"status": "error", "message": "System not available"})
        
        @self.app.route('/api/config')
        def api_config():
            """Get configuration"""
            return jsonify(self._get_config())
        
        @self.app.route('/api/config/update', methods=['POST'])
        def api_config_update():
            """Update configuration"""
            data = request.get_json()
            success = self._update_config(data)
            return jsonify({"status": "success" if success else "error"})
        
        @self.app.route('/api/logs')
        def api_logs():
            """Get recent logs"""
            return jsonify(self._get_recent_logs())
        
        @self.app.route('/api/health')
        def api_health():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime": self._get_uptime()
            })
    
    def _setup_socketio_events(self):
        """Setup SocketIO events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info("Client connected to dashboard")
            emit('status', self._get_system_status())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info("Client disconnected from dashboard")
        
        @self.socketio.on('request_update')
        def handle_request_update():
            """Handle update request"""
            emit('update', self._get_dashboard_data())
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            status = {
                "system_status": self.system_status,
                "is_running": self.is_running,
                "last_update": self.last_update.isoformat(),
                "uptime": self._get_uptime()
            }
            
            # Add component status
            if self.sniper_agent:
                status["sniper_agent"] = self.sniper_agent.get_status()
            
            if self.focus_agent:
                status["focus_agent"] = {
                    "is_running": self.focus_agent.is_running,
                    "performance_metrics": self.focus_agent.get_performance_metrics()
                }
            
            if self.risk_manager:
                status["risk_manager"] = {
                    "is_trading_allowed": self.risk_manager.is_trading_allowed(),
                    "performance_metrics": self.risk_manager.get_performance_metrics()
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat()
            }
            
            # Combine metrics from all components
            if self.sniper_agent:
                sniper_metrics = self.sniper_agent.performance_metrics
                metrics.update({
                    "total_opportunities": sniper_metrics.get("total_opportunities", 0),
                    "successful_trades": sniper_metrics.get("successful_trades", 0),
                    "failed_trades": sniper_metrics.get("failed_trades", 0),
                    "total_profit": sniper_metrics.get("total_profit", 0.0),
                    "ai_accuracy": sniper_metrics.get("ai_accuracy", 0.0)
                })
            
            if self.risk_manager:
                risk_metrics = self.risk_manager.get_performance_metrics()
                metrics.update({
                    "win_rate": risk_metrics.get("win_rate", 0.0),
                    "max_drawdown": risk_metrics.get("max_drawdown", 0.0),
                    "current_drawdown": risk_metrics.get("current_drawdown", 0.0),
                    "daily_pnl": risk_metrics.get("daily_pnl", 0.0)
                })
            
            if self.market_data:
                market_metrics = self.market_data.get_performance_metrics()
                metrics.update({
                    "api_call_count": market_metrics.get("api_call_count", 0),
                    "cache_hit_rate": market_metrics.get("cache_hit_rate", 0.0),
                    "error_count": market_metrics.get("error_count", 0)
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}
    
    def _get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            positions = []
            
            if self.sniper_agent and hasattr(self.sniper_agent, 'sniper_bot'):
                positions_data = asyncio.run(self.sniper_agent.sniper_bot.get_positions())
                
                for token_address, position in positions_data.items():
                    positions.append({
                        "token_address": token_address,
                        "amount": position.amount,
                        "average_price": position.average_price,
                        "current_price": position.current_price,
                        "unrealized_pnl": position.unrealized_pnl,
                        "entry_time": position.entry_time.isoformat(),
                        "last_updated": position.last_updated.isoformat()
                    })
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def _get_orders(self) -> List[Dict[str, Any]]:
        """Get order history"""
        try:
            orders = []
            
            if self.sniper_agent and hasattr(self.sniper_agent, 'sniper_bot'):
                orders_data = asyncio.run(self.sniper_agent.sniper_bot.get_order_history())
                
                for order in orders_data:
                    orders.append({
                        "order_id": order.order_id,
                        "token_address": order.token_address,
                        "action": order.action,
                        "amount": order.amount,
                        "price": order.price,
                        "status": order.status,
                        "created_at": order.created_at.isoformat(),
                        "executed_at": order.executed_at.isoformat() if order.executed_at else None,
                        "transaction_hash": order.transaction_hash,
                        "actual_price": order.actual_price,
                        "slippage": order.slippage
                    })
            
            return orders
            
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    def _get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics"""
        try:
            if self.risk_manager:
                portfolio_risk = asyncio.run(self.risk_manager.check_portfolio_risk())
                
                return {
                    "total_value": portfolio_risk.total_value,
                    "total_risk": portfolio_risk.total_risk,
                    "max_drawdown": portfolio_risk.max_drawdown,
                    "var_95": portfolio_risk.var_95,
                    "sharpe_ratio": portfolio_risk.sharpe_ratio,
                    "is_trading_allowed": self.risk_manager.is_trading_allowed(),
                    "circuit_breaker_active": self.risk_manager.circuit_breaker_active,
                    "risk_timestamp": portfolio_risk.risk_timestamp.isoformat()
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {"error": str(e)}
    
    def _get_market_data(self) -> Dict[str, Any]:
        """Get market data"""
        try:
            if self.market_data:
                market_data = asyncio.run(self.market_data.get_market_data())
                
                if market_data:
                    return {
                        "total_volume": market_data.total_volume,
                        "total_liquidity": market_data.total_liquidity,
                        "active_tokens": market_data.active_tokens,
                        "trending_tokens": market_data.trending_tokens,
                        "market_sentiment": market_data.market_sentiment,
                        "volatility_index": market_data.volatility_index,
                        "last_updated": market_data.last_updated.isoformat()
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {"error": str(e)}
    
    def _get_ai_status(self) -> Dict[str, Any]:
        """Get AI model status"""
        try:
            ai_status = {}
            
            if self.sniper_agent and hasattr(self.sniper_agent, 'model_factory'):
                model_factory = self.sniper_agent.model_factory
                ai_status.update({
                    "available_models": model_factory.get_available_models(),
                    "model_status": model_factory.get_model_status()
                })
            
            if self.chat_agent:
                chat_metrics = self.chat_agent.get_performance_metrics()
                ai_status.update({
                    "chat_metrics": chat_metrics,
                    "conversation_length": chat_metrics.get("conversation_length", 0)
                })
            
            return ai_status
            
        except Exception as e:
            logger.error(f"Error getting AI status: {e}")
            return {"error": str(e)}
    
    def _get_chat_history(self) -> List[Dict[str, Any]]:
        """Get chat history"""
        try:
            if self.chat_agent:
                return self.chat_agent.get_conversation_history()
            return []
            
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []
    
    async def _send_chat_message(self, message: str):
        """Send chat message"""
        try:
            if self.chat_agent:
                response = await self.chat_agent.add_user_message(message)
                
                # Emit response to connected clients
                self.socketio.emit('chat_response', {
                    "message": message,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error sending chat message: {e}")
    
    async def _restart_system(self):
        """Restart the system"""
        try:
            if self.sniper_agent:
                if self.sniper_agent.is_running:
                    await self.sniper_agent.stop()
                
                await asyncio.sleep(2)
                await self.sniper_agent.start()
                
        except Exception as e:
            logger.error(f"Error restarting system: {e}")
    
    def _get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        try:
            settings = get_settings()
            
            return {
                "ai_config": settings.ai_config.model_dump() if settings.ai_config else {},
                "trading_config": settings.trading_config.model_dump() if settings.trading_config else {},
                "system_config": {
                    "debug_mode": settings.debug_mode,
                    "log_level": settings.log_level,
                    "max_workers": settings.max_workers
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting config: {e}")
            return {"error": str(e)}
    
    def _update_config(self, config_data: Dict[str, Any]) -> bool:
        """Update configuration"""
        try:
            settings = get_settings()
            
            # Update AI config
            if "ai_config" in config_data:
                settings.ai_config = type(settings.ai_config)(**config_data["ai_config"])
                settings.save_ai_config()
            
            # Update trading config
            if "trading_config" in config_data:
                settings.trading_config = type(settings.trading_config)(**config_data["trading_config"])
                settings.save_trading_config()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return False
    
    def _get_recent_logs(self) -> List[Dict[str, Any]]:
        """Get recent logs"""
        try:
            # This would typically read from log files
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Error getting recent logs: {e}")
            return []
    
    def _get_uptime(self) -> str:
        """Get system uptime"""
        try:
            if hasattr(self, '_start_time'):
                uptime = datetime.now() - self._start_time
                return str(uptime).split('.')[0]  # Remove microseconds
            return "0:00:00"
            
        except Exception as e:
            logger.error(f"Error getting uptime: {e}")
            return "0:00:00"
    
    def _get_dashboard_data(self) -> Dict[str, Any]:
        """Get all dashboard data"""
        return {
            "status": self._get_system_status(),
            "performance": self._get_performance_metrics(),
            "positions": self._get_positions(),
            "orders": self._get_orders(),
            "risk": self._get_risk_metrics(),
            "market": self._get_market_data(),
            "ai": self._get_ai_status(),
            "chat": self._get_chat_history()
        }
    
    def set_components(self, sniper_agent=None, chat_agent=None, focus_agent=None, 
                      risk_manager=None, market_data=None):
        """Set system components"""
        self.sniper_agent = sniper_agent
        self.chat_agent = chat_agent
        self.focus_agent = focus_agent
        self.risk_manager = risk_manager
        self.market_data = market_data
    
    def start(self):
        """Start the dashboard"""
        if self.is_running:
            logger.warning("Dashboard is already running")
            return
        
        try:
            logger.info(f"Starting dashboard on {self.host}:{self.port}")
            self._start_time = datetime.now()
            self.is_running = True
            self.system_status = "running"
            
            # Start SocketIO server
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False
            )
            
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            self.is_running = False
            self.system_status = "error"
    
    def stop(self):
        """Stop the dashboard"""
        logger.info("Stopping dashboard")
        self.is_running = False
        self.system_status = "stopped"
    
    def emit_update(self, event: str, data: Any):
        """Emit update to connected clients"""
        if self.is_running:
            self.socketio.emit(event, data)
    
    def broadcast_status(self):
        """Broadcast current status to all connected clients"""
        if self.is_running:
            self.socketio.emit('status_update', self._get_system_status())


def create_dashboard(config: Dict[str, Any]) -> Dashboard:
    """Create dashboard instance"""
    return Dashboard(config)


if __name__ == "__main__":
    # Test dashboard
    config = {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 5000,
        "debug": True,
        "secret_key": "test-secret-key"
    }
    
    dashboard = create_dashboard(config)
    dashboard.start() 