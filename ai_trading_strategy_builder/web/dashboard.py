#!/usr/bin/env python3
"""
AI Trading Strategy Builder Web Dashboard

A comprehensive web interface for managing AI-generated trading strategies,
monitoring performance, and controlling strategy deployment.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_socketio import SocketIO, emit
import threading
import queue

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import AITradingStrategyBuilder


class TradingDashboard:
    """
    Web dashboard for the AI Trading Strategy Builder.
    """
    
    def __init__(self, strategy_builder: AITradingStrategyBuilder):
        self.strategy_builder = strategy_builder
        self.app = Flask(__name__)
        self.app.secret_key = 'your-secret-key-here'  # Change in production
        
        # Initialize SocketIO for real-time updates
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Setup routes
        self._setup_routes()
        
        # Background task queue
        self.task_queue = queue.Queue()
        self.running_tasks = {}
        
        # Start background task processor
        self._start_background_processor()
        
        self.logger = logging.getLogger("dashboard")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template('index.html')
        
        @self.app.route('/strategies')
        def strategies():
            """Strategies management page."""
            return render_template('strategies.html')
        
        @self.app.route('/backtesting')
        def backtesting():
            """Backtesting page."""
            return render_template('backtesting.html')
        
        @self.app.route('/deployment')
        def deployment():
            """Strategy deployment page."""
            return render_template('deployment.html')
        
        @self.app.route('/analytics')
        def analytics():
            """Analytics and performance page."""
            return render_template('analytics.html')
        
        @self.app.route('/api/strategies', methods=['GET'])
        def api_get_strategies():
            """API endpoint to get all strategies."""
            try:
                strategies = list(self.strategy_builder.strategies.values())
                return jsonify({
                    'success': True,
                    'strategies': strategies,
                    'total': len(strategies)
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/strategies', methods=['POST'])
        def api_create_strategy():
            """API endpoint to create a new strategy."""
            try:
                data = request.get_json()
                strategy_type = data.get('type')
                parameters = data.get('parameters', {})
                
                if not strategy_type:
                    return jsonify({
                        'success': False,
                        'error': 'Strategy type is required'
                    }), 400
                
                # Create strategy asynchronously
                asyncio.run(self.strategy_builder.generate_strategy(strategy_type, parameters))
                
                return jsonify({
                    'success': True,
                    'message': f'Strategy {strategy_type} created successfully'
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/strategies/<strategy_id>/backtest', methods=['POST'])
        def api_backtest_strategy(strategy_id):
            """API endpoint to backtest a strategy."""
            try:
                # Get strategy
                if strategy_id not in self.strategy_builder.strategies:
                    return jsonify({
                        'success': False,
                        'error': 'Strategy not found'
                    }), 404
                
                # Create sample data for backtesting
                sample_data = self.strategy_builder._create_sample_data()
                
                # Run backtest asynchronously
                asyncio.run(self.strategy_builder.backtest_strategy(strategy_id, sample_data))
                
                return jsonify({
                    'success': True,
                    'message': f'Strategy {strategy_id} backtested successfully'
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/strategies/<strategy_id>/deploy', methods=['POST'])
        def api_deploy_strategy(strategy_id):
            """API endpoint to deploy a strategy."""
            try:
                # Deploy strategy asynchronously
                asyncio.run(self.strategy_builder.deploy_strategy(strategy_id))
                
                return jsonify({
                    'success': True,
                    'message': f'Strategy {strategy_id} deployed successfully'
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/strategies/<strategy_id>', methods=['DELETE'])
        def api_delete_strategy(strategy_id):
            """API endpoint to delete a strategy."""
            try:
                if strategy_id in self.strategy_builder.strategies:
                    del self.strategy_builder.strategies[strategy_id]
                
                if strategy_id in self.strategy_builder.active_strategies:
                    del self.strategy_builder.active_strategies[strategy_id]
                
                return jsonify({
                    'success': True,
                    'message': f'Strategy {strategy_id} deleted successfully'
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/summary')
        def api_get_summary():
            """API endpoint to get system summary."""
            try:
                summary = self.strategy_builder.get_strategy_summary()
                return jsonify({
                    'success': True,
                    'summary': summary
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/performance/<strategy_id>')
        def api_get_performance(strategy_id):
            """API endpoint to get strategy performance."""
            try:
                if strategy_id not in self.strategy_builder.strategies:
                    return jsonify({
                        'success': False,
                        'error': 'Strategy not found'
                    }), 404
                
                strategy = self.strategy_builder.strategies[strategy_id]
                backtest_results = strategy.get('backtest_results', {})
                
                # Generate performance charts
                charts = self._generate_performance_charts(strategy_id, backtest_results)
                
                return jsonify({
                    'success': True,
                    'performance': backtest_results,
                    'charts': charts
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/health')
        def api_health_check():
            """API health check endpoint."""
            return jsonify({
                'success': True,
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            })
    
    def _start_background_processor(self):
        """Start background task processor."""
        def background_worker():
            while True:
                try:
                    task = self.task_queue.get(timeout=1)
                    if task is None:  # Shutdown signal
                        break
                    
                    task_id, func, args, kwargs = task
                    try:
                        result = func(*args, **kwargs)
                        self.running_tasks[task_id] = {
                            'status': 'completed',
                            'result': result,
                            'completed_at': datetime.now()
                        }
                        
                        # Emit completion event
                        self.socketio.emit('task_completed', {
                            'task_id': task_id,
                            'status': 'completed',
                            'result': result
                        })
                        
                    except Exception as e:
                        self.running_tasks[task_id] = {
                            'status': 'failed',
                            'error': str(e),
                            'failed_at': datetime.now()
                        }
                        
                        # Emit failure event
                        self.socketio.emit('task_failed', {
                            'task_id': task_id,
                            'status': 'failed',
                            'error': str(e)
                        })
                        
                except queue.Empty:
                    continue
        
        # Start background thread
        self.background_thread = threading.Thread(target=background_worker, daemon=True)
        self.background_thread.start()
    
    def _generate_performance_charts(self, strategy_id: str, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance charts for a strategy."""
        try:
            charts = {}
            
            # Create sample data for charts (in production, use actual backtest data)
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            np.random.seed(42)
            
            # Cumulative returns chart
            returns = np.random.normal(0.001, 0.02, len(dates))
            cumulative_returns = (1 + pd.Series(returns)).cumprod()
            
            fig_returns = go.Figure()
            fig_returns.add_trace(go.Scatter(
                x=dates,
                y=cumulative_returns,
                mode='lines',
                name='Cumulative Returns',
                line=dict(color='blue')
            ))
            fig_returns.update_layout(
                title=f'Cumulative Returns - {strategy_id}',
                xaxis_title='Date',
                yaxis_title='Cumulative Returns',
                template='plotly_white'
            )
            
            charts['cumulative_returns'] = json.dumps(fig_returns, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Drawdown chart
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            
            fig_drawdown = go.Figure()
            fig_drawdown.add_trace(go.Scatter(
                x=dates,
                y=drawdown * 100,
                mode='lines',
                name='Drawdown (%)',
                line=dict(color='red'),
                fill='tonexty'
            ))
            fig_drawdown.update_layout(
                title=f'Drawdown Analysis - {strategy_id}',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                template='plotly_white'
            )
            
            charts['drawdown'] = json.dumps(fig_drawdown, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Monthly returns heatmap
            monthly_returns = pd.Series(returns).groupby(pd.Grouper(freq='M')).apply(
                lambda x: (1 + x).prod() - 1
            )
            
            fig_monthly = go.Figure(data=go.Heatmap(
                z=monthly_returns.values.reshape(-1, 1),
                x=['Returns'],
                y=monthly_returns.index.strftime('%Y-%m'),
                colorscale='RdYlGn',
                text=monthly_returns.values.round(4),
                texttemplate='%{text:.2%}',
                textfont={"size": 10}
            ))
            fig_monthly.update_layout(
                title=f'Monthly Returns - {strategy_id}',
                template='plotly_white'
            )
            
            charts['monthly_returns'] = json.dumps(fig_monthly, cls=plotly.utils.PlotlyJSONEncoder)
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error generating performance charts: {e}")
            return {}
    
    def add_background_task(self, task_id: str, func, *args, **kwargs):
        """Add a background task to the queue."""
        self.running_tasks[task_id] = {
            'status': 'queued',
            'queued_at': datetime.now()
        }
        
        self.task_queue.put((task_id, func, args, kwargs))
        
        # Emit queued event
        self.socketio.emit('task_queued', {
            'task_id': task_id,
            'status': 'queued'
        })
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a background task."""
        return self.running_tasks.get(task_id)
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the dashboard."""
        self.logger.info(f"Starting AI Trading Dashboard on {host}:{port}")
        
        if debug:
            self.app.run(host=host, port=port, debug=debug)
        else:
            self.socketio.run(self.app, host=host, port=port, debug=debug)


# SocketIO event handlers
def setup_socketio_events(socketio, dashboard):
    """Setup SocketIO event handlers."""
    
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
        emit('connected', {'data': 'Connected to AI Trading Dashboard'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')
    
    @socketio.on('get_summary')
    def handle_get_summary():
        """Handle summary request."""
        try:
            summary = dashboard.strategy_builder.get_strategy_summary()
            emit('summary_update', summary)
        except Exception as e:
            emit('error', {'error': str(e)})
    
    @socketio.on('create_strategy')
    def handle_create_strategy(data):
        """Handle strategy creation request."""
        try:
            task_id = f"create_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            def create_strategy_task():
                return asyncio.run(
                    dashboard.strategy_builder.generate_strategy(
                        data['type'], 
                        data.get('parameters', {})
                    )
                )
            
            dashboard.add_background_task(task_id, create_strategy_task)
            emit('task_created', {'task_id': task_id})
            
        except Exception as e:
            emit('error', {'error': str(e)})
    
    @socketio.on('backtest_strategy')
    def handle_backtest_strategy(data):
        """Handle strategy backtesting request."""
        try:
            strategy_id = data['strategy_id']
            task_id = f"backtest_{strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            def backtest_strategy_task():
                sample_data = dashboard.strategy_builder._create_sample_data()
                return asyncio.run(
                    dashboard.strategy_builder.backtest_strategy(strategy_id, sample_data)
                )
            
            dashboard.add_background_task(task_id, backtest_strategy_task)
            emit('task_created', {'task_id': task_id})
            
        except Exception as e:
            emit('error', {'error': str(e)})


def create_dashboard(strategy_builder: AITradingStrategyBuilder) -> TradingDashboard:
    """Create and configure the trading dashboard."""
    dashboard = TradingDashboard(strategy_builder)
    
    # Setup SocketIO events
    setup_socketio_events(dashboard.socketio, dashboard)
    
    return dashboard


if __name__ == "__main__":
    # Create a sample strategy builder for testing
    async def create_sample_builder():
        builder = AITradingStrategyBuilder()
        return builder
    
    # Run the dashboard
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    strategy_builder = loop.run_until_complete(create_sample_builder())
    
    dashboard = create_dashboard(strategy_builder)
    dashboard.run(debug=True)
