"""
AI Trading Algorithm Development Platform - Main Entry Point
Implements the RBI (Research, Backtest, Implement) methodology for algorithmic trading.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from research.market_analyzer import MarketAnalyzer
from research.strategy_discovery import StrategyDiscovery
from backtesting.backtest_engine import BacktestEngine, BacktestConfig
from implementation.live_trading_engine import LiveTradingEngine
from ai_tools.code_generator import CodeGenerator
from ai_tools.strategy_builder import StrategyBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_platform.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Trading Algorithm Development Platform",
    description="RBI methodology for systematic algorithmic trading development",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for platform components
market_analyzer = None
strategy_discovery = None
code_generator = None
strategy_builder = None


@app.on_event("startup")
async def startup_event():
    """Initialize platform on startup."""
    global market_analyzer, strategy_discovery, code_generator, strategy_builder
    
    logger.info("🚀 Starting AI Trading Algorithm Development Platform...")
    logger.info("📚 Mission: RBI methodology for systematic trading development")
    logger.info("🎯 Focus: Research, Backtest, Implement - not price prediction")
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    Path("strategies").mkdir(exist_ok=True)
    Path("backtests").mkdir(exist_ok=True)
    
    # Load configuration
    config = load_configuration()
    
    # Initialize platform components
    market_analyzer = MarketAnalyzer(config)
    strategy_discovery = StrategyDiscovery(config)
    code_generator = CodeGenerator(config)
    strategy_builder = StrategyBuilder(config)
    
    logger.info("✅ AI Trading Algorithm Platform initialized successfully")


def load_configuration() -> Dict[str, Any]:
    """Load platform configuration."""
    try:
        import json
        with open("config/api_keys.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("Configuration file not found, using default config")
        return {
            "openai": {"api_key": "your_openai_api_key"},
            "anthropic": {"api_key": "your_anthropic_api_key"},
            "yfinance": {"enabled": True},
            "alpha_vantage": {"api_key": "your_alpha_vantage_key"},
            "polygon": {"api_key": "your_polygon_key"}
        }


@app.get("/")
async def root():
    """Platform root endpoint."""
    return {
        "message": "Welcome to AI Trading Algorithm Development Platform",
        "mission": "RBI methodology for systematic trading development",
        "philosophy": "Research, Backtest, Implement - not price prediction",
        "version": "1.0.0",
        "status": "active",
        "methodology": "RBI (Research, Backtest, Implement)"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "market_analyzer": "active",
            "strategy_discovery": "active",
            "code_generator": "active",
            "strategy_builder": "active"
        },
        "methodology": "RBI - Research, Backtest, Implement"
    }


# Research Phase Endpoints
@app.post("/api/research/market-analysis")
async def analyze_market(analysis_request: Dict[str, Any]):
    """Analyze market opportunities using AI."""
    global market_analyzer
    
    try:
        symbols = analysis_request.get("symbols", ["SPY"])
        timeframe = analysis_request.get("timeframe", "1y")
        analysis_type = analysis_request.get("analysis_type", "comprehensive")
        
        analysis = await market_analyzer.analyze_market(
            symbols=symbols,
            timeframe=timeframe,
            analysis_type=analysis_type
        )
        
        return {
            "analysis": analysis,
            "opportunities": analysis.get("opportunities", []),
            "risks": analysis.get("risks", []),
            "recommendations": analysis.get("recommendations", [])
        }
        
    except Exception as e:
        logger.error(f"Error analyzing market: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/research/strategy-discovery")
async def discover_strategies(discovery_request: Dict[str, Any]):
    """Discover promising trading strategies."""
    global strategy_discovery
    
    try:
        market_data = discovery_request.get("market_data", {})
        strategy_types = discovery_request.get("strategy_types", ["momentum", "mean_reversion"])
        
        strategies = await strategy_discovery.discover_strategies(
            market_data=market_data,
            strategy_types=strategy_types
        )
        
        return {
            "strategies": strategies,
            "strategy_count": len(strategies),
            "recommended_strategies": [s for s in strategies if s.get("recommended", False)]
        }
        
    except Exception as e:
        logger.error(f"Error discovering strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/research/opportunity-finder")
async def find_opportunities(opportunity_request: Dict[str, Any]):
    """Find specific trading opportunities."""
    global market_analyzer
    
    try:
        symbols = opportunity_request.get("symbols", ["SPY"])
        opportunity_type = opportunity_request.get("type", "momentum")
        
        opportunities = await market_analyzer.find_opportunities(
            symbols=symbols,
            opportunity_type=opportunity_type
        )
        
        return {
            "opportunities": opportunities,
            "opportunity_count": len(opportunities),
            "best_opportunities": [o for o in opportunities if o.get("score", 0) > 0.7]
        }
        
    except Exception as e:
        logger.error(f"Error finding opportunities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Backtest Phase Endpoints
@app.post("/api/backtest/run")
async def run_backtest(backtest_request: Dict[str, Any]):
    """Run comprehensive backtest following RBI methodology."""
    try:
        # Create backtest configuration
        config = BacktestConfig(
            start_date=backtest_request["start_date"],
            end_date=backtest_request["end_date"],
            initial_capital=backtest_request.get("initial_capital", 100000.0),
            commission=backtest_request.get("commission", 0.001),
            slippage=backtest_request.get("slippage", 0.0005),
            symbols=backtest_request.get("symbols", ["SPY"])
        )
        
        # Create backtest engine
        engine = BacktestEngine(config)
        
        # Get strategy function
        strategy_name = backtest_request.get("strategy", "momentum")
        strategy_params = backtest_request.get("strategy_params", {})
        
        # Import strategy function
        if strategy_name == "momentum":
            from backtesting.backtest_engine import momentum_strategy
            strategy_func = momentum_strategy
        elif strategy_name == "mean_reversion":
            from backtesting.backtest_engine import mean_reversion_strategy
            strategy_func = mean_reversion_strategy
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Run comprehensive backtest
        result = engine.run_comprehensive_backtest(strategy_func, strategy_params)
        
        # Generate report
        report = engine.generate_comprehensive_report(result)
        
        return {
            "backtest_id": f"backtest_{int(result.start_date.timestamp())}",
            "strategy_name": result.strategy_name,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "walk_forward_stable": result.walk_forward_results.get("is_stable", False) if result.walk_forward_results else False,
            "out_of_sample_robust": result.out_of_sample_results.get("is_robust", False) if result.out_of_sample_results else False,
            "report": report,
            "recommendation": generate_backtest_recommendation(result)
        }
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest/optimize")
async def optimize_strategy(optimization_request: Dict[str, Any]):
    """Optimize strategy parameters."""
    try:
        # Create backtest configuration
        config = BacktestConfig(
            start_date=optimization_request["start_date"],
            end_date=optimization_request["end_date"],
            initial_capital=optimization_request.get("initial_capital", 100000.0),
            symbols=optimization_request.get("symbols", ["SPY"])
        )
        
        # Create backtest engine
        engine = BacktestEngine(config)
        
        # Get strategy and parameter ranges
        strategy_name = optimization_request.get("strategy", "momentum")
        param_ranges = optimization_request.get("param_ranges", {})
        
        # Import strategy function
        if strategy_name == "momentum":
            from backtesting.backtest_engine import momentum_strategy
            strategy_func = momentum_strategy
        elif strategy_name == "mean_reversion":
            from backtesting.backtest_engine import mean_reversion_strategy
            strategy_func = mean_reversion_strategy
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Run parameter optimization
        optimization_results = []
        
        # Grid search optimization
        for lookback in param_ranges.get("lookback", [10, 20, 30, 50]):
            for threshold in param_ranges.get("threshold", [0.01, 0.02, 0.03, 0.05]):
                params = {"lookback": lookback, "threshold": threshold}
                
                try:
                    result = engine.run_comprehensive_backtest(strategy_func, params)
                    
                    optimization_results.append({
                        "parameters": params,
                        "sharpe_ratio": result.sharpe_ratio,
                        "total_return": result.total_return,
                        "max_drawdown": result.max_drawdown,
                        "win_rate": result.win_rate,
                        "is_stable": result.walk_forward_results.get("is_stable", False) if result.walk_forward_results else False,
                        "is_robust": result.out_of_sample_results.get("is_robust", False) if result.out_of_sample_results else False
                    })
                except Exception as e:
                    logger.warning(f"Error testing parameters {params}: {e}")
        
        # Find best parameters
        if optimization_results:
            # Filter for stable and robust strategies
            stable_results = [r for r in optimization_results if r["is_stable"] and r["is_robust"]]
            
            if stable_results:
                best_result = max(stable_results, key=lambda x: x["sharpe_ratio"])
            else:
                best_result = max(optimization_results, key=lambda x: x["sharpe_ratio"])
            
            return {
                "optimization_results": optimization_results,
                "best_parameters": best_result["parameters"],
                "best_sharpe": best_result["sharpe_ratio"],
                "best_return": best_result["total_return"],
                "best_drawdown": best_result["max_drawdown"],
                "stable_strategies": len([r for r in optimization_results if r["is_stable"]]),
                "robust_strategies": len([r for r in optimization_results if r["is_robust"]])
            }
        else:
            return {"error": "No valid optimization results found"}
        
    except Exception as e:
        logger.error(f"Error optimizing strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Implementation Phase Endpoints
@app.post("/api/implementation/validate")
async def validate_for_implementation(validation_request: Dict[str, Any]):
    """Validate if a strategy is ready for live implementation."""
    try:
        backtest_results = validation_request.get("backtest_results", {})
        
        # Validation criteria
        criteria = {
            "min_sharpe_ratio": 1.0,
            "max_drawdown": 0.20,
            "min_trades": 30,
            "min_win_rate": 0.40,
            "min_profit_factor": 1.2
        }
        
        validation_results = {
            "sharpe_ratio_ok": backtest_results.get("sharpe_ratio", 0) >= criteria["min_sharpe_ratio"],
            "drawdown_ok": backtest_results.get("max_drawdown", 1) <= criteria["max_drawdown"],
            "trades_ok": backtest_results.get("total_trades", 0) >= criteria["min_trades"],
            "win_rate_ok": backtest_results.get("win_rate", 0) >= criteria["min_win_rate"],
            "profit_factor_ok": backtest_results.get("profit_factor", 0) >= criteria["min_profit_factor"],
            "walk_forward_stable": backtest_results.get("walk_forward_stable", False),
            "out_of_sample_robust": backtest_results.get("out_of_sample_robust", False)
        }
        
        # Overall validation
        all_criteria_met = all(validation_results.values())
        
        # Generate recommendations
        recommendations = []
        if not validation_results["sharpe_ratio_ok"]:
            recommendations.append("Sharpe ratio below 1.0 - consider strategy improvement")
        if not validation_results["drawdown_ok"]:
            recommendations.append("Maximum drawdown too high - implement stricter risk management")
        if not validation_results["trades_ok"]:
            recommendations.append("Insufficient trades - test on longer time period")
        if not validation_results["walk_forward_stable"]:
            recommendations.append("Strategy not stable in walk-forward analysis - risk of overfitting")
        if not validation_results["out_of_sample_robust"]:
            recommendations.append("Strategy not robust in out-of-sample testing")
        
        if all_criteria_met:
            recommendations.append("Strategy ready for live implementation with small position sizes")
        
        return {
            "ready_for_implementation": all_criteria_met,
            "validation_results": validation_results,
            "recommendations": recommendations,
            "risk_level": "low" if all_criteria_met else "high"
        }
        
    except Exception as e:
        logger.error(f"Error validating for implementation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/implementation/start-live")
async def start_live_trading(live_request: Dict[str, Any]):
    """Start live trading with validated strategy."""
    try:
        strategy_config = live_request.get("strategy_config", {})
        position_size = live_request.get("position_size", 0.01)  # 1% default
        risk_limits = live_request.get("risk_limits", {})
        
        # Validate strategy first
        validation_result = await validate_for_implementation({
            "backtest_results": strategy_config.get("backtest_results", {})
        })
        
        if not validation_result["ready_for_implementation"]:
            raise HTTPException(status_code=400, detail="Strategy not ready for live implementation")
        
        # Create live trading engine
        from implementation.live_trading_engine import LiveTradingEngine
        live_engine = LiveTradingEngine(strategy_config, position_size, risk_limits)
        
        # Start live trading
        trading_id = await live_engine.start_trading()
        
        return {
            "trading_id": trading_id,
            "status": "started",
            "position_size": position_size,
            "risk_limits": risk_limits,
            "message": "Live trading started successfully"
        }
        
    except Exception as e:
        logger.error(f"Error starting live trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# AI Tools Endpoints
@app.post("/api/ai/generate-code")
async def generate_strategy_code(code_request: Dict[str, Any]):
    """Generate strategy code using AI."""
    global code_generator
    
    try:
        strategy_description = code_request.get("description", "")
        strategy_type = code_request.get("type", "momentum")
        parameters = code_request.get("parameters", {})
        
        code = await code_generator.generate_strategy_code(
            description=strategy_description,
            strategy_type=strategy_type,
            parameters=parameters
        )
        
        return {
            "code": code,
            "strategy_type": strategy_type,
            "parameters": parameters,
            "message": "Strategy code generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error generating code: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/build-strategy")
async def build_visual_strategy(strategy_request: Dict[str, Any]):
    """Build strategy using visual AI builder."""
    global strategy_builder
    
    try:
        strategy_components = strategy_request.get("components", [])
        strategy_connections = strategy_request.get("connections", [])
        
        strategy = await strategy_builder.build_strategy(
            components=strategy_components,
            connections=strategy_connections
        )
        
        return {
            "strategy_id": strategy.get("id"),
            "code": strategy.get("code"),
            "validation": strategy.get("validation"),
            "message": "Strategy built successfully"
        }
        
    except Exception as e:
        logger.error(f"Error building strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Strategy Library Endpoints
@app.get("/api/strategies")
async def get_strategy_library():
    """Get available strategies in the library."""
    try:
        strategies = [
            {
                "id": "momentum_basic",
                "name": "Basic Momentum Strategy",
                "type": "momentum",
                "description": "Simple momentum strategy based on price trends",
                "parameters": {"lookback": 20, "threshold": 0.02},
                "category": "beginner"
            },
            {
                "id": "mean_reversion_bb",
                "name": "Bollinger Bands Mean Reversion",
                "type": "mean_reversion",
                "description": "Mean reversion strategy using Bollinger Bands",
                "parameters": {"bb_period": 20, "bb_std": 2},
                "category": "intermediate"
            },
            {
                "id": "vwap_strategy",
                "name": "VWAP Strategy",
                "type": "mean_reversion",
                "description": "Volume Weighted Average Price strategy",
                "parameters": {"vwap_period": 20},
                "category": "intermediate"
            },
            {
                "id": "rsi_divergence",
                "name": "RSI Divergence Strategy",
                "type": "mean_reversion",
                "description": "RSI divergence detection strategy",
                "parameters": {"rsi_period": 14, "divergence_threshold": 0.1},
                "category": "advanced"
            }
        ]
        
        return {
            "strategies": strategies,
            "total_count": len(strategies),
            "categories": ["beginner", "intermediate", "advanced"]
        }
        
    except Exception as e:
        logger.error(f"Error getting strategy library: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategies/{strategy_id}")
async def get_strategy_details(strategy_id: str):
    """Get detailed information about a specific strategy."""
    try:
        # In a real implementation, this would fetch from a database
        strategy_details = {
            "id": strategy_id,
            "name": f"Strategy {strategy_id}",
            "description": "Detailed strategy description",
            "code": "# Strategy code would be here",
            "parameters": {"param1": "value1"},
            "performance": {
                "avg_sharpe": 1.5,
                "avg_return": 0.15,
                "avg_drawdown": 0.10
            }
        }
        
        return strategy_details
        
    except Exception as e:
        logger.error(f"Error getting strategy details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Performance Analytics Endpoints
@app.get("/api/analytics/performance")
async def get_performance_analytics():
    """Get platform performance analytics."""
    try:
        analytics = {
            "total_backtests": 150,
            "successful_strategies": 45,
            "live_trading_strategies": 12,
            "average_sharpe_ratio": 1.2,
            "average_return": 0.18,
            "average_drawdown": 0.12,
            "top_performing_strategy": {
                "name": "Advanced Momentum",
                "sharpe_ratio": 2.1,
                "return": 0.25
            }
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting performance analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Educational Resources Endpoints
@app.get("/api/education/resources")
async def get_educational_resources():
    """Get educational resources for algorithmic trading."""
    try:
        resources = {
            "courses": [
                {
                    "name": "Stanford Machine Learning Course",
                    "instructor": "Andrew Ng",
                    "url": "https://www.coursera.org/learn/machine-learning",
                    "description": "Comprehensive machine learning fundamentals"
                },
                {
                    "name": "MIT Algorithmic Trading",
                    "instructor": "MIT",
                    "url": "https://ocw.mit.edu/courses/15-450-analytics-of-finance-fall-2010/",
                    "description": "MIT's algorithmic trading course"
                }
            ],
            "books": [
                {
                    "title": "Advances in Financial Machine Learning",
                    "author": "Marcos Lopez de Prado",
                    "description": "Advanced ML applications in finance"
                },
                {
                    "title": "Building Algorithmic Trading Systems",
                    "author": "Kevin Davey",
                    "description": "Practical guide to building trading systems"
                }
            ],
            "papers": [
                {
                    "title": "Renaissance Technologies Papers",
                    "description": "Academic papers from successful quant firm"
                }
            ]
        }
        
        return resources
        
    except Exception as e:
        logger.error(f"Error getting educational resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def generate_backtest_recommendation(result: Any) -> str:
    """Generate recommendation based on backtest results."""
    if result.sharpe_ratio >= 1.5 and result.max_drawdown <= 0.15:
        return "Excellent strategy - ready for live implementation"
    elif result.sharpe_ratio >= 1.0 and result.max_drawdown <= 0.20:
        return "Good strategy - consider implementation with small position sizes"
    elif result.sharpe_ratio >= 0.5:
        return "Moderate strategy - needs improvement before implementation"
    else:
        return "Poor strategy - not recommended for implementation"


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 