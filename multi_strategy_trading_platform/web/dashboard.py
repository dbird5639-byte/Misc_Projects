"""
Web Dashboard for Multi-Strategy Trading Platform
FastAPI-based dashboard for monitoring and control
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def create_dashboard(
    strategies: Dict[str, Any],
    test_results: Dict[str, Any],
    portfolio_performance: Dict[str, Any],
    config: Dict[str, Any]
) -> FastAPI:
    """Create FastAPI dashboard application"""
    
    app = FastAPI(title="Multi-Strategy Trading Platform Dashboard")
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Main dashboard page"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multi-Strategy Trading Platform</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e8f4f8; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Multi-Strategy Trading Platform Dashboard</h1>
                <p>Built on the wisdom of Jacob Amaral and Kevin Davy</p>
            </div>
            
            <div class="section">
                <h2>Platform Status</h2>
                <div class="metric"><strong>Strategies:</strong> {len(strategies)}</div>
                <div class="metric"><strong>Test Results:</strong> {len(test_results)}</div>
                <div class="metric"><strong>Portfolio Active:</strong> {'Yes' if portfolio_performance else 'No'}</div>
            </div>
            
            <div class="section">
                <h2>Configuration</h2>
                <div class="metric"><strong>Max Drawdown:</strong> {config.get('max_drawdown', 0):.1%}</div>
                <div class="metric"><strong>Risk Per Trade:</strong> {config.get('risk_per_trade', 0):.1%}</div>
                <div class="metric"><strong>Max Correlation:</strong> {config.get('max_correlation', 0):.1f}</div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html)
    
    @app.get("/api/strategies")
    async def get_strategies():
        """Get strategies data"""
        return {"strategies": list(strategies.keys())}
    
    @app.get("/api/test-results")
    async def get_test_results():
        """Get test results data"""
        return {"test_results": list(test_results.keys())}
    
    @app.get("/api/portfolio")
    async def get_portfolio():
        """Get portfolio performance data"""
        return {"portfolio": portfolio_performance}
    
    return app 