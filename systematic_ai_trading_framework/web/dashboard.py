"""
Web Dashboard for the Systematic AI Trading Framework.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO, emit
import pandas as pd
import plotly.graph_objs as go
import plotly.utils

from config.settings import Settings
from utils.logger import setup_logger


class TradingDashboard:
    """
    Web dashboard for monitoring the systematic AI trading framework.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = setup_logger("dashboard", settings.log_level)
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'systematic-ai-trading-secret-key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Dashboard state
        self.framework_status = {
            'is_running': False,
            'active_strategies': 0,
            'discovered_strategies': 0,
            'backtest_results': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'risk_level': 'low'
        }
        
        self.strategies_data = {}
        self.performance_data = {}
        self.ai_agent_status = {}
        
        # Set up routes
        self._setup_routes()
        self._setup_socketio_events()
        
        self.logger.info("Trading Dashboard initialized")
    
    def _setup_routes(self):
        """Set up Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template('dashboard.html', 
                                framework_status=self.framework_status,
                                strategies_data=self.strategies_data)
        
        @self.app.route('/api/status')
        def api_status():
            """API endpoint for framework status."""
            return jsonify(self.framework_status)
        
        @self.app.route('/api/strategies')
        def api_strategies():
            """API endpoint for strategies data."""
            return jsonify(self.strategies_data)
        
        @self.app.route('/api/performance')
        def api_performance():
            """API endpoint for performance data."""
            return jsonify(self.performance_data)
        
        @self.app.route('/api/ai_agents')
        def api_ai_agents():
            """API endpoint for AI agent status."""
            return jsonify(self.ai_agent_status)
        
        @self.app.route('/api/equity_chart')
        def api_equity_chart():
            """API endpoint for equity curve chart."""
            chart_data = self._generate_equity_chart()
            return jsonify(chart_data)
        
        @self.app.route('/api/performance_chart')
        def api_performance_chart():
            """API endpoint for performance chart."""
            chart_data = self._generate_performance_chart()
            return jsonify(chart_data)
        
        @self.app.route('/api/strategy_details/<strategy_name>')
        def api_strategy_details(strategy_name):
            """API endpoint for detailed strategy information."""
            if strategy_name in self.strategies_data:
                return jsonify(self.strategies_data[strategy_name])
            return jsonify({'error': 'Strategy not found'}), 404
        
        @self.app.route('/api/backtest_results/<strategy_name>')
        def api_backtest_results(strategy_name):
            """API endpoint for backtest results."""
            # This would fetch actual backtest results
            mock_results = self._generate_mock_backtest_results(strategy_name)
            return jsonify(mock_results)
        
        @self.app.route('/api/control/start', methods=['POST'])
        def api_start_framework():
            """API endpoint to start the framework."""
            try:
                # This would actually start the framework
                self.framework_status['is_running'] = True
                self.framework_status['last_started'] = datetime.now().isoformat()
                self.socketio.emit('framework_status_update', self.framework_status)
                return jsonify({'status': 'success', 'message': 'Framework started'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/control/stop', methods=['POST'])
        def api_stop_framework():
            """API endpoint to stop the framework."""
            try:
                # This would actually stop the framework
                self.framework_status['is_running'] = False
                self.framework_status['last_stopped'] = datetime.now().isoformat()
                self.socketio.emit('framework_status_update', self.framework_status)
                return jsonify({'status': 'success', 'message': 'Framework stopped'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/control/deploy_strategy', methods=['POST'])
        def api_deploy_strategy():
            """API endpoint to deploy a strategy."""
            try:
                data = request.get_json()
                strategy_name = data.get('strategy_name')
                
                if strategy_name:
                    # This would actually deploy the strategy
                    self.framework_status['active_strategies'] += 1
                    self.socketio.emit('strategy_deployed', {
                        'strategy_name': strategy_name,
                        'timestamp': datetime.now().isoformat()
                    })
                    return jsonify({'status': 'success', 'message': f'Strategy {strategy_name} deployed'})
                else:
                    return jsonify({'status': 'error', 'message': 'Strategy name required'}), 400
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def _setup_socketio_events(self):
        """Set up SocketIO events for real-time updates."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            self.logger.info("Client connected to dashboard")
            emit('framework_status_update', self.framework_status)
            emit('strategies_update', self.strategies_data)
            emit('performance_update', self.performance_data)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            self.logger.info("Client disconnected from dashboard")
        
        @self.socketio.on('request_update')
        def handle_update_request():
            """Handle update requests from clients."""
            emit('framework_status_update', self.framework_status)
            emit('strategies_update', self.strategies_data)
            emit('performance_update', self.performance_data)
    
    def update_framework_status(self, status: Dict[str, Any]):
        """Update framework status and broadcast to clients."""
        self.framework_status.update(status)
        self.socketio.emit('framework_status_update', self.framework_status)
    
    def update_strategy_data(self, strategy_name: str, data: Dict[str, Any]):
        """Update strategy data and broadcast to clients."""
        self.strategies_data[strategy_name] = data
        self.socketio.emit('strategy_update', {
            'strategy_name': strategy_name,
            'data': data
        })
    
    def update_performance_data(self, data: Dict[str, Any]):
        """Update performance data and broadcast to clients."""
        self.performance_data.update(data)
        self.socketio.emit('performance_update', self.performance_data)
    
    def update_ai_agent_status(self, agent_name: str, status: Dict[str, Any]):
        """Update AI agent status and broadcast to clients."""
        self.ai_agent_status[agent_name] = status
        self.socketio.emit('ai_agent_update', {
            'agent_name': agent_name,
            'status': status
        })
    
    def _generate_equity_chart(self) -> Dict[str, Any]:
        """Generate equity curve chart data."""
        # Mock data - in real implementation, this would use actual equity data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                            end=datetime.now(), freq='D')
        equity_values = [100000 + i * 1000 + np.random.normal(0, 500) for i in range(len(dates))]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#2E86AB', width=2)
        ))
        
        fig.update_layout(
            title='Portfolio Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            template='plotly_white',
            height=400
        )
        
        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
    
    def _generate_performance_chart(self) -> Dict[str, Any]:
        """Generate performance comparison chart."""
        # Mock data for strategy performance comparison
        strategies = ['Momentum Strategy', 'Mean Reversion', 'Regime Detection']
        returns = [0.15, 0.12, 0.18]
        sharpe_ratios = [1.85, 1.42, 2.01]
        max_drawdowns = [0.08, 0.12, 0.06]
        
        fig = go.Figure()
        
        # Returns bar chart
        fig.add_trace(go.Bar(
            x=strategies,
            y=returns,
            name='Total Return (%)',
            marker_color='#A23B72'
        ))
        
        fig.update_layout(
            title='Strategy Performance Comparison',
            xaxis_title='Strategy',
            yaxis_title='Total Return (%)',
            template='plotly_white',
            height=400,
            barmode='group'
        )
        
        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
    
    def _generate_mock_backtest_results(self, strategy_name: str) -> Dict[str, Any]:
        """Generate mock backtest results for a strategy."""
        return {
            'strategy_name': strategy_name,
            'backtest_period': {
                'start': '2023-01-01',
                'end': '2024-01-01'
            },
            'performance_metrics': {
                'total_return': 0.15,
                'sharpe_ratio': 1.85,
                'max_drawdown': 0.08,
                'win_rate': 0.62,
                'profit_factor': 1.75,
                'total_trades': 156,
                'avg_trade_duration': 3.2
            },
            'risk_metrics': {
                'var_95': -0.02,
                'cvar_95': -0.03,
                'volatility': 0.12,
                'beta': 0.85
            },
            'trade_analysis': {
                'largest_win': 0.05,
                'largest_loss': -0.03,
                'avg_win': 0.02,
                'avg_loss': -0.015,
                'consecutive_wins': 8,
                'consecutive_losses': 3
            }
        }
    
    def start(self, host: str = '0.0.0.0', port: int = 8080, debug: bool = False):
        """Start the dashboard server."""
        self.logger.info(f"Starting dashboard on {host}:{port}")
        
        # Start with mock data for demonstration
        self._initialize_mock_data()
        
        # Start the server
        self.socketio.run(self.app, host=host, port=port, debug=debug)
    
    def _initialize_mock_data(self):
        """Initialize dashboard with mock data for demonstration."""
        # Framework status
        self.framework_status = {
            'is_running': True,
            'active_strategies': 3,
            'discovered_strategies': 8,
            'backtest_results': 12,
            'total_pnl': 15430.50,
            'daily_pnl': 2340.25,
            'risk_level': 'medium',
            'last_started': datetime.now().isoformat(),
            'uptime': '2 days, 5 hours'
        }
        
        # Strategies data
        self.strategies_data = {
            'momentum_strategy': {
                'name': 'Momentum Strategy',
                'status': 'active',
                'pnl': 8230.50,
                'pnl_pct': 0.082,
                'sharpe_ratio': 1.85,
                'max_drawdown': 0.082,
                'win_rate': 0.62,
                'total_trades': 45,
                'open_positions': 2,
                'last_signal': '2024-01-15T10:30:00Z',
                'parameters': {
                    'lookback_period': 20,
                    'threshold': 0.02,
                    'rsi_period': 14
                }
            },
            'mean_reversion_strategy': {
                'name': 'Mean Reversion Strategy',
                'status': 'active',
                'pnl': 4560.25,
                'pnl_pct': 0.046,
                'sharpe_ratio': 1.42,
                'max_drawdown': 0.121,
                'win_rate': 0.58,
                'total_trades': 38,
                'open_positions': 1,
                'last_signal': '2024-01-15T09:45:00Z',
                'parameters': {
                    'bb_period': 20,
                    'bb_std': 2,
                    'rsi_period': 14
                }
            },
            'regime_detection_strategy': {
                'name': 'Regime Detection Strategy',
                'status': 'active',
                'pnl': 2639.75,
                'pnl_pct': 0.026,
                'sharpe_ratio': 2.01,
                'max_drawdown': 0.068,
                'win_rate': 0.65,
                'total_trades': 28,
                'open_positions': 3,
                'last_signal': '2024-01-15T11:15:00Z',
                'parameters': {
                    'volatility_period': 20,
                    'correlation_period': 60,
                    'regime_threshold': 0.7
                }
            }
        }
        
        # Performance data
        self.performance_data = {
            'total_portfolio_value': 115430.50,
            'daily_return': 0.021,
            'weekly_return': 0.045,
            'monthly_return': 0.082,
            'annual_return': 0.154,
            'volatility': 0.12,
            'sharpe_ratio': 1.76,
            'sortino_ratio': 2.34,
            'max_drawdown': 0.082,
            'calmar_ratio': 1.88,
            'risk_free_rate': 0.02
        }
        
        # AI agent status
        self.ai_agent_status = {
            'research_agent': {
                'status': 'active',
                'last_activity': '2024-01-15T11:30:00Z',
                'discovered_ideas': 15,
                'processing_time': '2.3s',
                'model': 'deepseek-coder:6.7b'
            },
            'backtest_agent': {
                'status': 'active',
                'last_activity': '2024-01-15T11:25:00Z',
                'strategies_tested': 3,
                'processing_time': '45.2s',
                'model': 'deepseek-coder:6.7b'
            },
            'package_agent': {
                'status': 'idle',
                'last_activity': '2024-01-15T10:15:00Z',
                'dependencies_checked': 156,
                'processing_time': '1.1s',
                'model': 'llama2:7b'
            }
        }


async def start_dashboard(port: int = 8080):
    """Start the trading dashboard."""
    settings = Settings()
    dashboard = TradingDashboard(settings)
    dashboard.start(port=port)


if __name__ == "__main__":
    asyncio.run(start_dashboard()) 