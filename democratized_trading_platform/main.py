"""
Democratized Trading Platform - Main Entry Point
Making algorithmic trading accessible to everyone, not just geniuses.
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

from education.learning_path import LearningPath
from backtesting.engine import BacktestEngine, BacktestConfig
from strategy_builder.visual_builder import VisualStrategyBuilder
from community.mentorship import MentorshipSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/platform.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Democratized Trading Platform",
    description="Making algorithmic trading accessible to everyone, not just geniuses",
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

# Initialize platform components
learning_path = LearningPath()
mentorship_system = MentorshipSystem()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize platform on startup."""
    logger.info("🚀 Starting Democratized Trading Platform...")
    logger.info("📚 Mission: Making algorithmic trading accessible to everyone")
    logger.info("🎯 Framework: RBI (Research, Backtest, Implement)")
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    
    logger.info("✅ Platform initialized successfully")


@app.get("/")
async def root():
    """Platform root endpoint."""
    return {
        "message": "Welcome to the Democratized Trading Platform",
        "mission": "Making algorithmic trading accessible to everyone, not just geniuses",
        "framework": "RBI (Research, Backtest, Implement)",
        "version": "1.0.0",
        "status": "active"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "learning_path": "active",
            "mentorship": "active",
            "backtesting": "active",
            "strategy_builder": "active"
        }
    }


# Learning Path Endpoints
@app.get("/api/learning-path/progress/{user_id}")
async def get_learning_progress(user_id: str):
    """Get user's learning progress."""
    try:
        progress = learning_path.get_user_progress(user_id)
        next_module = learning_path.get_next_module(user_id)
        recommendations = learning_path.get_recommended_resources(user_id)
        
        return {
            "user_id": user_id,
            "progress": progress,
            "next_module": {
                "id": next_module.id,
                "title": next_module.title,
                "description": next_module.description,
                "duration_minutes": next_module.duration_minutes
            } if next_module else None,
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Error getting learning progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/learning-path/progress/{user_id}")
async def update_learning_progress(user_id: str, module_id: str, progress: float):
    """Update user's learning progress."""
    try:
        learning_path.update_progress(user_id, module_id, progress)
        return {"message": "Progress updated successfully"}
    except Exception as e:
        logger.error(f"Error updating learning progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/learning-path/plan/{user_id}")
async def generate_learning_plan(user_id: str, target_date: str):
    """Generate a personalized learning plan."""
    try:
        from datetime import datetime
        target_datetime = datetime.fromisoformat(target_date)
        plan = learning_path.generate_learning_plan(user_id, target_datetime)
        return plan
    except Exception as e:
        logger.error(f"Error generating learning plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Backtesting Endpoints
@app.post("/api/backtest/run")
async def run_backtest(config: Dict[str, Any]):
    """Run a backtest with the provided configuration."""
    try:
        # Convert config to BacktestConfig
        backtest_config = BacktestConfig(
            start_date=config["start_date"],
            end_date=config["end_date"],
            initial_capital=config.get("initial_capital", 100000.0),
            commission=config.get("commission", 0.001),
            slippage=config.get("slippage", 0.0005),
            symbols=config.get("symbols", ["SPY"])
        )
        
        # Create backtest engine
        engine = BacktestEngine(backtest_config)
        
        # Get strategy function from config
        strategy_name = config.get("strategy", "momentum")
        if strategy_name == "momentum":
            from backtesting.engine import momentum_strategy
            strategy_func = momentum_strategy
            strategy_params = config.get("strategy_params", {"lookback": 20, "threshold": 0.02})
        elif strategy_name == "mean_reversion":
            from backtesting.engine import mean_reversion_strategy
            strategy_func = mean_reversion_strategy
            strategy_params = config.get("strategy_params", {"bb_period": 20, "bb_std": 2})
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Run backtest
        result = engine.run_backtest(strategy_func, strategy_params)
        
        # Convert result to dict for JSON serialization
        result_dict = {
            "strategy_name": result.strategy_name,
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "calmar_ratio": result.calmar_ratio,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "total_trades": result.total_trades,
            "avg_trade_duration": result.avg_trade_duration,
            "performance_metrics": result.performance_metrics,
            "risk_metrics": result.risk_metrics
        }
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/report/{strategy_name}")
async def generate_backtest_report(strategy_name: str, config: Dict[str, Any]):
    """Generate a comprehensive backtest report."""
    try:
        # Run backtest first
        backtest_result = await run_backtest(config)
        
        # Generate report
        from backtesting.engine import BacktestEngine
        engine = BacktestEngine(BacktestConfig(
            start_date=config["start_date"],
            end_date=config["end_date"]
        ))
        
        # Create a mock result object for report generation
        class MockResult:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        mock_result = MockResult(backtest_result)
        report = engine.generate_report(mock_result)
        
        return {"report": report, "results": backtest_result}
        
    except Exception as e:
        logger.error(f"Error generating backtest report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Strategy Builder Endpoints
@app.post("/api/strategy-builder/create")
async def create_strategy(strategy_data: Dict[str, Any]):
    """Create a new visual strategy."""
    try:
        builder = VisualStrategyBuilder()
        builder.strategy_name = strategy_data.get("name", "My Strategy")
        builder.description = strategy_data.get("description", "")
        
        # Add components
        for component_data in strategy_data.get("components", []):
            component_type = component_data["type"]
            
            if component_type == "indicator":
                from strategy_builder.visual_builder import Indicator, IndicatorType, ComponentType
                component = Indicator(
                    id=component_data["id"],
                    name=component_data["name"],
                    component_type=ComponentType.INDICATOR,
                    indicator_type=IndicatorType(component_data["indicator_type"]),
                    parameters=component_data.get("parameters", {}),
                    output_name=component_data.get("output_name", "")
                )
            
            elif component_type == "condition":
                from strategy_builder.visual_builder import Condition, ConditionType, ComponentType
                component = Condition(
                    id=component_data["id"],
                    name=component_data["name"],
                    component_type=ComponentType.CONDITION,
                    condition_type=ConditionType(component_data["condition_type"]),
                    left_input=component_data.get("left_input", ""),
                    right_input=component_data.get("right_input", ""),
                    threshold=component_data.get("threshold", 0.0)
                )
            
            elif component_type == "action":
                from strategy_builder.visual_builder import Action, ActionType, ComponentType
                component = Action(
                    id=component_data["id"],
                    name=component_data["name"],
                    component_type=ComponentType.ACTION,
                    action_type=ActionType(component_data["action_type"]),
                    quantity=component_data.get("quantity", 1.0)
                )
            
            builder.add_component(component)
        
        # Connect components
        for connection in strategy_data.get("connections", []):
            builder.connect_components(connection["from"], connection["to"])
        
        # Validate strategy
        validation = builder.validate_strategy()
        
        # Generate Python code
        code = builder.generate_python_code()
        
        return {
            "strategy_id": strategy_data.get("id", "generated_strategy"),
            "validation": validation,
            "python_code": code,
            "message": "Strategy created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/strategy-builder/backtest")
async def backtest_visual_strategy(strategy_data: Dict[str, Any], config: Dict[str, Any]):
    """Run backtest on a visual strategy."""
    try:
        # Create strategy first
        strategy_result = await create_strategy(strategy_data)
        
        if not strategy_result["validation"]["is_valid"]:
            raise HTTPException(status_code=400, detail="Strategy is not valid")
        
        # Create backtest config
        backtest_config = BacktestConfig(
            start_date=config["start_date"],
            end_date=config["end_date"],
            initial_capital=config.get("initial_capital", 100000.0),
            commission=config.get("commission", 0.001),
            slippage=config.get("slippage", 0.0005),
            symbols=config.get("symbols", ["SPY"])
        )
        
        # Create builder and run backtest
        builder = VisualStrategyBuilder()
        result = builder.run_backtest(backtest_config)
        
        # Convert result to dict
        result_dict = {
            "strategy_name": result.strategy_name,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades
        }
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error backtesting visual strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mentorship Endpoints
@app.post("/api/mentorship/register-mentor")
async def register_mentor(mentor_data: Dict[str, Any]):
    """Register a new mentor."""
    try:
        mentor_id = mentorship_system.register_mentor(mentor_data)
        return {"mentor_id": mentor_id, "message": "Mentor registered successfully"}
    except Exception as e:
        logger.error(f"Error registering mentor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mentorship/register-mentee")
async def register_mentee(mentee_data: Dict[str, Any]):
    """Register a new mentee."""
    try:
        mentee_id = mentorship_system.register_mentee(mentee_data)
        return {"mentee_id": mentee_id, "message": "Mentee registered successfully"}
    except Exception as e:
        logger.error(f"Error registering mentee: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mentorship/recommendations/{mentee_id}")
async def get_mentor_recommendations(mentee_id: str):
    """Get mentor recommendations for a mentee."""
    try:
        recommendations = mentorship_system.get_recommended_mentors(mentee_id)
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error getting mentor recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mentorship/request")
async def request_mentorship(request_data: Dict[str, Any]):
    """Request a mentorship relationship."""
    try:
        relationship_id = mentorship_system.request_mentorship(
            request_data["mentee_id"],
            request_data["mentor_id"],
            request_data["goals"]
        )
        return {"relationship_id": relationship_id, "message": "Mentorship requested"}
    except Exception as e:
        logger.error(f"Error requesting mentorship: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mentorship/schedule-session")
async def schedule_mentorship_session(session_data: Dict[str, Any]):
    """Schedule a mentorship session."""
    try:
        from datetime import datetime
        from community.mentorship import SessionType
        
        session_time = datetime.fromisoformat(session_data["scheduled_time"])
        session_type = SessionType(session_data["session_type"])
        
        session_id = mentorship_system.schedule_session(
            session_data["relationship_id"],
            session_type,
            session_time,
            session_data.get("duration_minutes", 60)
        )
        
        return {"session_id": session_id, "message": "Session scheduled"}
    except Exception as e:
        logger.error(f"Error scheduling session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Community Endpoints
@app.get("/api/community/stats")
async def get_community_stats():
    """Get community statistics."""
    try:
        # Get basic stats
        total_mentors = len(mentorship_system.mentors)
        total_mentees = len(mentorship_system.mentees)
        total_sessions = len(mentorship_system.sessions)
        active_relationships = len([
            r for r in mentorship_system.relationships.values()
            if r.status.value == "active"
        ])
        
        return {
            "total_mentors": total_mentors,
            "total_mentees": total_mentees,
            "total_sessions": total_sessions,
            "active_relationships": active_relationships,
            "platform_status": "active"
        }
    except Exception as e:
        logger.error(f"Error getting community stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/community/success-stories")
async def get_success_stories():
    """Get community success stories."""
    try:
        # Mock success stories - in a real implementation, these would come from a database
        stories = [
            {
                "id": "1",
                "name": "John Smith",
                "age": 35,
                "background": "No coding experience",
                "achievement": "Built a momentum strategy generating $2,000/month",
                "time_to_success": "6 months",
                "quote": "The platform made algorithmic trading accessible to me. I started with zero coding experience and now have a profitable strategy running."
            },
            {
                "id": "2",
                "name": "Sarah Johnson",
                "age": 28,
                "background": "Finance degree, no programming",
                "achievement": "Mean reversion strategy with 65% win rate and 1.8 Sharpe ratio",
                "time_to_success": "8 months",
                "quote": "The community and mentorship program helped me avoid common pitfalls. The systematic approach really works."
            },
            {
                "id": "3",
                "name": "Mike Chen",
                "age": 42,
                "background": "Software engineer, no trading experience",
                "achievement": "3 automated strategies running with $5,000/month total income",
                "time_to_success": "12 months",
                "quote": "I was intimidated by algorithmic trading until I found this platform. The RBI framework is brilliant."
            }
        ]
        
        return {"stories": stories}
    except Exception as e:
        logger.error(f"Error getting success stories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Educational Resources Endpoints
@app.get("/api/education/resources")
async def get_educational_resources():
    """Get educational resources."""
    try:
        resources = {
            "books": [
                {
                    "title": "Building Algorithmic Trading Systems",
                    "author": "Kevin Davey",
                    "category": "beginner",
                    "description": "Comprehensive guide to building trading systems"
                },
                {
                    "title": "Python for Finance",
                    "author": "Yves Hilpisch",
                    "category": "beginner",
                    "description": "Learn Python for financial applications"
                },
                {
                    "title": "Advances in Financial Machine Learning",
                    "author": "Marcos Lopez de Prado",
                    "category": "advanced",
                    "description": "Advanced machine learning for finance"
                }
            ],
            "videos": [
                {
                    "title": "Introduction to Algorithmic Trading",
                    "duration": "45 minutes",
                    "category": "beginner",
                    "url": "/videos/intro-algo-trading.mp4"
                },
                {
                    "title": "Python Basics for Trading",
                    "duration": "120 minutes",
                    "category": "beginner",
                    "url": "/videos/python-basics.mp4"
                },
                {
                    "title": "Backtesting Fundamentals",
                    "duration": "60 minutes",
                    "category": "intermediate",
                    "url": "/videos/backtesting-basics.mp4"
                }
            ],
            "articles": [
                {
                    "title": "What is Algorithmic Trading?",
                    "category": "beginner",
                    "url": "/articles/what-is-algo-trading.md"
                },
                {
                    "title": "The RBI Framework Explained",
                    "category": "beginner",
                    "url": "/articles/rbi-framework.md"
                },
                {
                    "title": "Risk Management for Beginners",
                    "category": "intermediate",
                    "url": "/articles/risk-management.md"
                }
            ]
        }
        
        return resources
    except Exception as e:
        logger.error(f"Error getting educational resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 