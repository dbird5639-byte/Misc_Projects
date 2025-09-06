"""
Structured Learning Path for Democratized Trading Platform
Guides users from beginner to advanced algorithmic trading.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class SkillLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ContentType(Enum):
    VIDEO = "video"
    READING = "reading"
    EXERCISE = "exercise"
    PROJECT = "project"
    QUIZ = "quiz"
    WORKSHOP = "workshop"


@dataclass
class LearningModule:
    """Represents a learning module in the curriculum."""
    id: str
    title: str
    description: str
    skill_level: SkillLevel
    content_type: ContentType
    duration_minutes: int
    prerequisites: List[str] = field(default_factory=list)
    resources: List[Dict[str, str]] = field(default_factory=list)
    exercises: List[Dict[str, Any]] = field(default_factory=list)
    completed: bool = False
    progress: float = 0.0


@dataclass
class LearningPhase:
    """Represents a learning phase (weeks 1-4, 5-8, etc.)."""
    id: str
    title: str
    description: str
    duration_weeks: int
    modules: List[LearningModule]
    objectives: List[str]
    assessment: Dict[str, Any]


class LearningPath:
    """
    Structured learning path for algorithmic trading education.
    Based on the RBI framework: Research, Backtest, Implement.
    """
    
    def __init__(self):
        self.phases = self._create_learning_phases()
        self.current_user_progress = {}
        self.assessments = {}
        
    def _create_learning_phases(self) -> List[LearningPhase]:
        """Create the complete learning curriculum."""
        return [
            self._create_phase_1_foundation(),
            self._create_phase_2_research(),
            self._create_phase_3_backtesting(),
            self._create_phase_4_implementation()
        ]
    
    def _create_phase_1_foundation(self) -> LearningPhase:
        """Phase 1: Foundation (Weeks 1-4)"""
        modules = [
            LearningModule(
                id="intro_algo_trading",
                title="Introduction to Algorithmic Trading",
                description="Understanding what algorithmic trading is and why it's accessible to everyone",
                skill_level=SkillLevel.BEGINNER,
                content_type=ContentType.VIDEO,
                duration_minutes=45,
                resources=[
                    {"type": "video", "url": "/videos/intro-algo-trading.mp4"},
                    {"type": "reading", "url": "/articles/what-is-algo-trading.md"},
                    {"type": "book", "title": "Building Algorithmic Trading Systems", "author": "Kevin Davey"}
                ],
                exercises=[
                    {"type": "quiz", "title": "Algo Trading Basics", "questions": 10},
                    {"type": "discussion", "title": "Why Algo Trading?", "prompt": "Share your motivation"}
                ]
            ),
            LearningModule(
                id="python_basics",
                title="Python Programming Fundamentals",
                description="Learn Python basics needed for algorithmic trading",
                skill_level=SkillLevel.BEGINNER,
                content_type=ContentType.VIDEO,
                duration_minutes=120,
                prerequisites=["intro_algo_trading"],
                resources=[
                    {"type": "video", "url": "/videos/python-basics.mp4"},
                    {"type": "interactive", "url": "/exercises/python-basics"},
                    {"type": "book", "title": "Python for Finance", "author": "Yves Hilpisch"}
                ],
                exercises=[
                    {"type": "coding", "title": "Basic Python Exercises", "tasks": 15},
                    {"type": "project", "title": "Simple Calculator", "description": "Build a basic calculator"}
                ]
            ),
            LearningModule(
                id="market_data",
                title="Understanding Market Data",
                description="Learn about OHLCV data and market structure",
                skill_level=SkillLevel.BEGINNER,
                content_type=ContentType.VIDEO,
                duration_minutes=60,
                prerequisites=["python_basics"],
                resources=[
                    {"type": "video", "url": "/videos/market-data.mp4"},
                    {"type": "dataset", "url": "/data/sample_market_data.csv"},
                    {"type": "article", "url": "/articles/understanding-ohlcv.md"}
                ],
                exercises=[
                    {"type": "data_analysis", "title": "Analyze Sample Data", "dataset": "sample_market_data.csv"},
                    {"type": "quiz", "title": "Market Data Quiz", "questions": 8}
                ]
            ),
            LearningModule(
                id="first_strategy",
                title="Your First Trading Strategy",
                description="Build and understand your first simple strategy",
                skill_level=SkillLevel.BEGINNER,
                content_type=ContentType.PROJECT,
                duration_minutes=90,
                prerequisites=["market_data"],
                resources=[
                    {"type": "video", "url": "/videos/first-strategy.mp4"},
                    {"type": "template", "url": "/templates/simple_momentum.py"},
                    {"type": "article", "url": "/articles/momentum-strategy.md"}
                ],
                exercises=[
                    {"type": "coding", "title": "Build Momentum Strategy", "template": "simple_momentum.py"},
                    {"type": "discussion", "title": "Strategy Discussion", "prompt": "Share your strategy insights"}
                ]
            )
        ]
        
        return LearningPhase(
            id="phase_1",
            title="Foundation",
            description="Build the fundamental knowledge and skills needed for algorithmic trading",
            duration_weeks=4,
            modules=modules,
            objectives=[
                "Understand what algorithmic trading is and why it's accessible",
                "Learn basic Python programming skills",
                "Understand market data structure and sources",
                "Build and understand your first trading strategy"
            ],
            assessment={
                "type": "comprehensive_quiz",
                "questions": 25,
                "passing_score": 80,
                "practical_project": "Build a simple momentum strategy"
            }
        )
    
    def _create_phase_2_research(self) -> LearningPhase:
        """Phase 2: Research (Weeks 5-8)"""
        modules = [
            LearningModule(
                id="strategy_research",
                title="Strategy Research Methods",
                description="Learn how to research and evaluate trading strategies",
                skill_level=SkillLevel.INTERMEDIATE,
                content_type=ContentType.VIDEO,
                duration_minutes=75,
                prerequisites=["first_strategy"],
                resources=[
                    {"type": "video", "url": "/videos/strategy-research.mp4"},
                    {"type": "article", "url": "/articles/research-methodology.md"},
                    {"type": "database", "url": "/research/strategy-library"}
                ],
                exercises=[
                    {"type": "research", "title": "Strategy Analysis", "strategies": ["momentum", "mean_reversion", "breakout"]},
                    {"type": "discussion", "title": "Research Findings", "prompt": "Share your research insights"}
                ]
            ),
            LearningModule(
                id="academic_papers",
                title="Reading Academic Papers",
                description="Learn to read and understand academic trading research",
                skill_level=SkillLevel.INTERMEDIATE,
                content_type=ContentType.READING,
                duration_minutes=60,
                prerequisites=["strategy_research"],
                resources=[
                    {"type": "video", "url": "/videos/reading-papers.mp4"},
                    {"type": "papers", "url": "/papers/curated-list.md"},
                    {"type": "guide", "url": "/guides/paper-reading-guide.md"}
                ],
                exercises=[
                    {"type": "paper_analysis", "title": "Paper Review", "papers": ["momentum_1993", "mean_reversion_1990"]},
                    {"type": "summary", "title": "Paper Summary", "format": "executive_summary"}
                ]
            ),
            LearningModule(
                id="market_analysis",
                title="Market Analysis Tools",
                description="Learn to analyze markets and identify opportunities",
                skill_level=SkillLevel.INTERMEDIATE,
                content_type=ContentType.VIDEO,
                duration_minutes=90,
                prerequisites=["academic_papers"],
                resources=[
                    {"type": "video", "url": "/videos/market-analysis.mp4"},
                    {"type": "tools", "url": "/tools/market-scanner"},
                    {"type": "article", "url": "/articles/market-analysis.md"}
                ],
                exercises=[
                    {"type": "analysis", "title": "Market Scan", "markets": ["SPY", "QQQ", "IWM"]},
                    {"type": "report", "title": "Market Report", "format": "analysis_report"}
                ]
            ),
            LearningModule(
                id="strategy_selection",
                title="Strategy Selection and Evaluation",
                description="Learn to select and evaluate strategies for implementation",
                skill_level=SkillLevel.INTERMEDIATE,
                content_type=ContentType.PROJECT,
                duration_minutes=120,
                prerequisites=["market_analysis"],
                resources=[
                    {"type": "video", "url": "/videos/strategy-selection.mp4"},
                    {"type": "framework", "url": "/frameworks/strategy-evaluation.md"},
                    {"type": "template", "url": "/templates/strategy-evaluation.py"}
                ],
                exercises=[
                    {"type": "evaluation", "title": "Strategy Evaluation", "strategies": 5},
                    {"type": "presentation", "title": "Strategy Presentation", "format": "pitch_deck"}
                ]
            )
        ]
        
        return LearningPhase(
            id="phase_2",
            title="Research",
            description="Learn to research, analyze, and select trading strategies",
            duration_weeks=4,
            modules=modules,
            objectives=[
                "Master strategy research methodology",
                "Learn to read and understand academic papers",
                "Use market analysis tools effectively",
                "Select and evaluate strategies for implementation"
            ],
            assessment={
                "type": "research_project",
                "deliverable": "strategy_evaluation_report",
                "presentation": True,
                "peer_review": True
            }
        )
    
    def _create_phase_3_backtesting(self) -> LearningPhase:
        """Phase 3: Backtesting (Weeks 9-12)"""
        modules = [
            LearningModule(
                id="backtesting_fundamentals",
                title="Backtesting Fundamentals",
                description="Learn the basics of backtesting and why it's crucial",
                skill_level=SkillLevel.INTERMEDIATE,
                content_type=ContentType.VIDEO,
                duration_minutes=60,
                prerequisites=["strategy_selection"],
                resources=[
                    {"type": "video", "url": "/videos/backtesting-basics.mp4"},
                    {"type": "article", "url": "/articles/backtesting-importance.md"},
                    {"type": "book", "title": "Advances in Financial Machine Learning", "author": "Marcos Lopez de Prado"}
                ],
                exercises=[
                    {"type": "quiz", "title": "Backtesting Basics", "questions": 12},
                    {"type": "discussion", "title": "Backtesting Importance", "prompt": "Why is backtesting crucial?"}
                ]
            ),
            LearningModule(
                id="performance_metrics",
                title="Performance Metrics and Analysis",
                description="Learn to calculate and interpret trading performance metrics",
                skill_level=SkillLevel.INTERMEDIATE,
                content_type=ContentType.VIDEO,
                duration_minutes=90,
                prerequisites=["backtesting_fundamentals"],
                resources=[
                    {"type": "video", "url": "/videos/performance-metrics.mp4"},
                    {"type": "calculator", "url": "/tools/metrics-calculator"},
                    {"type": "article", "url": "/articles/performance-metrics.md"}
                ],
                exercises=[
                    {"type": "calculation", "title": "Calculate Metrics", "metrics": ["sharpe", "drawdown", "win_rate"]},
                    {"type": "analysis", "title": "Performance Analysis", "dataset": "sample_results.csv"}
                ]
            ),
            LearningModule(
                id="walk_forward",
                title="Walk-Forward Analysis",
                description="Learn to prevent overfitting with walk-forward testing",
                skill_level=SkillLevel.ADVANCED,
                content_type=ContentType.VIDEO,
                duration_minutes=75,
                prerequisites=["performance_metrics"],
                resources=[
                    {"type": "video", "url": "/videos/walk-forward.mp4"},
                    {"type": "article", "url": "/articles/overfitting-prevention.md"},
                    {"type": "template", "url": "/templates/walk_forward.py"}
                ],
                exercises=[
                    {"type": "implementation", "title": "Walk-Forward Test", "strategy": "momentum"},
                    {"type": "analysis", "title": "Overfitting Analysis", "comparison": "in_sample_vs_out_of_sample"}
                ]
            ),
            LearningModule(
                id="strategy_validation",
                title="Strategy Validation and Optimization",
                description="Validate strategies and optimize parameters",
                skill_level=SkillLevel.ADVANCED,
                content_type=ContentType.PROJECT,
                duration_minutes=120,
                prerequisites=["walk_forward"],
                resources=[
                    {"type": "video", "url": "/videos/strategy-validation.mp4"},
                    {"type": "framework", "url": "/frameworks/validation.md"},
                    {"type": "tools", "url": "/tools/optimization-engine"}
                ],
                exercises=[
                    {"type": "optimization", "title": "Parameter Optimization", "method": "genetic_algorithm"},
                    {"type": "validation", "title": "Strategy Validation", "tests": ["robustness", "stability"]}
                ]
            )
        ]
        
        return LearningPhase(
            id="phase_3",
            title="Backtesting",
            description="Master backtesting, performance analysis, and strategy validation",
            duration_weeks=4,
            modules=modules,
            objectives=[
                "Understand backtesting fundamentals and importance",
                "Calculate and interpret performance metrics",
                "Implement walk-forward analysis to prevent overfitting",
                "Validate and optimize strategies effectively"
            ],
            assessment={
                "type": "backtesting_project",
                "deliverable": "comprehensive_backtest_report",
                "validation_required": True,
                "peer_review": True
            }
        )
    
    def _create_phase_4_implementation(self) -> LearningPhase:
        """Phase 4: Implementation (Weeks 13-16)"""
        modules = [
            LearningModule(
                id="risk_management",
                title="Risk Management Fundamentals",
                description="Learn essential risk management principles",
                skill_level=SkillLevel.INTERMEDIATE,
                content_type=ContentType.VIDEO,
                duration_minutes=60,
                prerequisites=["strategy_validation"],
                resources=[
                    {"type": "video", "url": "/videos/risk-management.mp4"},
                    {"type": "article", "url": "/articles/risk-management.md"},
                    {"type": "book", "title": "Risk Management for Traders", "author": "Bennett McDowell"}
                ],
                exercises=[
                    {"type": "calculation", "title": "Position Sizing", "methods": ["kelly", "fixed_fractional"]},
                    {"type": "simulation", "title": "Risk Simulation", "scenarios": ["market_crash", "volatility_spike"]}
                ]
            ),
            LearningModule(
                id="live_trading_setup",
                title="Live Trading Setup",
                description="Set up your first live trading system",
                skill_level=SkillLevel.ADVANCED,
                content_type=ContentType.PROJECT,
                duration_minutes=90,
                prerequisites=["risk_management"],
                resources=[
                    {"type": "video", "url": "/videos/live-setup.mp4"},
                    {"type": "guide", "url": "/guides/live-trading-setup.md"},
                    {"type": "template", "url": "/templates/live_trading.py"}
                ],
                exercises=[
                    {"type": "setup", "title": "Paper Trading", "duration": "2_weeks"},
                    {"type": "monitoring", "title": "System Monitoring", "tools": ["dashboard", "alerts"]}
                ]
            ),
            LearningModule(
                id="monitoring_adjustment",
                title="Monitoring and Adjustment",
                description="Monitor live strategies and make necessary adjustments",
                skill_level=SkillLevel.ADVANCED,
                content_type=ContentType.VIDEO,
                duration_minutes=75,
                prerequisites=["live_trading_setup"],
                resources=[
                    {"type": "video", "url": "/videos/monitoring.mp4"},
                    {"type": "dashboard", "url": "/tools/monitoring-dashboard"},
                    {"type": "article", "url": "/articles/strategy-monitoring.md"}
                ],
                exercises=[
                    {"type": "monitoring", "title": "Live Monitoring", "duration": "1_week"},
                    {"type": "adjustment", "title": "Strategy Adjustment", "scenarios": ["underperformance", "overperformance"]}
                ]
            ),
            LearningModule(
                id="scaling_optimization",
                title="Scaling and Optimization",
                description="Scale successful strategies and optimize performance",
                skill_level=SkillLevel.ADVANCED,
                content_type=ContentType.PROJECT,
                duration_minutes=120,
                prerequisites=["monitoring_adjustment"],
                resources=[
                    {"type": "video", "url": "/videos/scaling.mp4"},
                    {"type": "framework", "url": "/frameworks/scaling.md"},
                    {"type": "tools", "url": "/tools/portfolio-optimizer"}
                ],
                exercises=[
                    {"type": "scaling", "title": "Strategy Scaling", "methods": ["capital", "markets", "timeframes"]},
                    {"type": "optimization", "title": "Portfolio Optimization", "objective": "maximize_sharpe"}
                ]
            )
        ]
        
        return LearningPhase(
            id="phase_4",
            title="Implementation",
            description="Implement live trading systems with proper risk management",
            duration_weeks=4,
            modules=modules,
            objectives=[
                "Master risk management principles",
                "Set up and run live trading systems",
                "Monitor and adjust strategies effectively",
                "Scale and optimize successful strategies"
            ],
            assessment={
                "type": "live_trading_project",
                "deliverable": "live_trading_report",
                "paper_trading_required": True,
                "risk_management_audit": True
            }
        )
    
    def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        """Get user's learning progress."""
        if user_id not in self.current_user_progress:
            self.current_user_progress[user_id] = {
                "current_phase": 0,
                "current_module": 0,
                "completed_modules": [],
                "start_date": datetime.now(),
                "total_progress": 0.0
            }
        
        return self.current_user_progress[user_id]
    
    def update_progress(self, user_id: str, module_id: str, progress: float):
        """Update user's progress on a specific module."""
        if user_id not in self.current_user_progress:
            self.get_user_progress(user_id)
        
        # Find and update module progress
        for phase in self.phases:
            for module in phase.modules:
                if module.id == module_id:
                    module.progress = progress
                    if progress >= 100.0:
                        module.completed = True
                        if module_id not in self.current_user_progress[user_id]["completed_modules"]:
                            self.current_user_progress[user_id]["completed_modules"].append(module_id)
        
        # Update total progress
        self._calculate_total_progress(user_id)
    
    def _calculate_total_progress(self, user_id: str):
        """Calculate total progress across all phases."""
        total_modules = sum(len(phase.modules) for phase in self.phases)
        completed_modules = len(self.current_user_progress[user_id]["completed_modules"])
        
        self.current_user_progress[user_id]["total_progress"] = (completed_modules / total_modules) * 100
    
    def get_next_module(self, user_id: str) -> Optional[LearningModule]:
        """Get the next module for the user to complete."""
        progress = self.get_user_progress(user_id)
        
        for phase_idx, phase in enumerate(self.phases):
            for module_idx, module in enumerate(phase.modules):
                if not module.completed:
                    # Check prerequisites
                    if self._check_prerequisites(user_id, module.prerequisites):
                        return module
        
        return None
    
    def _check_prerequisites(self, user_id: str, prerequisites: List[str]) -> bool:
        """Check if user has completed all prerequisites."""
        completed_modules = self.current_user_progress[user_id]["completed_modules"]
        return all(prereq in completed_modules for prereq in prerequisites)
    
    def get_recommended_resources(self, user_id: str) -> List[Dict[str, str]]:
        """Get personalized resource recommendations based on user progress."""
        progress = self.get_user_progress(user_id)
        recommendations = []
        
        # Get current module
        current_module = self.get_next_module(user_id)
        if current_module:
            recommendations.extend(current_module.resources)
        
        # Add community resources
        recommendations.extend([
            {"type": "community", "url": "/community/forums"},
            {"type": "mentorship", "url": "/community/mentorship"},
            {"type": "workshop", "url": "/events/upcoming-workshops"}
        ])
        
        return recommendations
    
    def generate_learning_plan(self, user_id: str, target_completion_date: datetime) -> Dict[str, Any]:
        """Generate a personalized learning plan for the user."""
        progress = self.get_user_progress(user_id)
        days_remaining = (target_completion_date - datetime.now()).days
        
        # Calculate required pace
        total_modules = sum(len(phase.modules) for phase in self.phases)
        completed_modules = len(progress["completed_modules"])
        remaining_modules = total_modules - completed_modules
        
        if days_remaining <= 0:
            return {"error": "Target date has passed"}
        
        modules_per_day = remaining_modules / days_remaining
        
        return {
            "target_date": target_completion_date,
            "days_remaining": days_remaining,
            "modules_remaining": remaining_modules,
            "required_pace": f"{modules_per_day:.2f} modules per day",
            "recommended_schedule": self._generate_schedule(user_id, modules_per_day),
            "milestones": self._generate_milestones(user_id, target_completion_date)
        }
    
    def _generate_schedule(self, user_id: str, modules_per_day: float) -> List[Dict[str, Any]]:
        """Generate a daily learning schedule."""
        schedule = []
        current_date = datetime.now()
        
        for i in range(7):  # Next 7 days
            date = current_date + timedelta(days=i)
            modules_for_day = []
            
            # Get modules for this day
            for _ in range(int(modules_per_day)):
                module = self.get_next_module(user_id)
                if module:
                    modules_for_day.append({
                        "id": module.id,
                        "title": module.title,
                        "duration_minutes": module.duration_minutes
                    })
            
            schedule.append({
                "date": date.strftime("%Y-%m-%d"),
                "modules": modules_for_day,
                "total_time_minutes": sum(m["duration_minutes"] for m in modules_for_day)
            })
        
        return schedule
    
    def _generate_milestones(self, user_id: str, target_date: datetime) -> List[Dict[str, Any]]:
        """Generate learning milestones."""
        progress = self.get_user_progress(user_id)
        completed_modules = len(progress["completed_modules"])
        total_modules = sum(len(phase.modules) for phase in self.phases)
        
        milestones = []
        for i, phase in enumerate(self.phases):
            phase_completion = (completed_modules / total_modules) * 100
            milestones.append({
                "phase": phase.title,
                "description": f"Complete {phase.title} phase",
                "target_percentage": ((i + 1) / len(self.phases)) * 100,
                "current_percentage": phase_completion,
                "status": "completed" if phase_completion >= ((i + 1) / len(self.phases)) * 100 else "pending"
            })
        
        return milestones


# Example usage
if __name__ == "__main__":
    learning_path = LearningPath()
    
    # Get user progress
    user_progress = learning_path.get_user_progress("user123")
    print(f"User progress: {user_progress}")
    
    # Get next module
    next_module = learning_path.get_next_module("user123")
    if next_module:
        print(f"Next module: {next_module.title}")
    
    # Generate learning plan
    target_date = datetime.now() + timedelta(weeks=16)
    plan = learning_path.generate_learning_plan("user123", target_date)
    print(f"Learning plan: {plan}") 