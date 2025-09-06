"""
Quantitative Trading Research Platform
Main Entry Point

Comprehensive platform for systematic quantitative trading research,
inspired by Jim Simons and other top algorithmic traders.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import platform components
from research_engine.anomaly_detector import MarketAnomalyDetector, AnomalyType
from machine_learning.model_factory import ModelFactory, ModelType, FeatureType
from backtesting_framework.multi_market_backtester import MultiMarketBacktester
from data_management.market_data_collector import MarketDataCollector
from data_management.feature_engineer import FeatureEngineer
from strategy_library.mean_reversion import MeanReversionStrategy
from strategy_library.momentum import MomentumStrategy
from risk_management.portfolio_optimizer import PortfolioOptimizer
from research_tools.hypothesis_generator import HypothesisGenerator
from research_tools.experiment_designer import ExperimentDesigner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantitativeTradingResearchPlatform:
    """
    Main platform class for quantitative trading research
    
    Implements Jim Simons' approach of finding many small edges,
    building complex evolving models, and systematic testing.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the platform"""
        self.config = self._load_config(config_path)
        
        # Initialize core components
        self.anomaly_detector = MarketAnomalyDetector()
        self.model_factory = ModelFactory()
        self.data_collector = MarketDataCollector()
        self.feature_engineer = FeatureEngineer()
        self.backtester = MultiMarketBacktester()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.hypothesis_generator = HypothesisGenerator()
        self.experiment_designer = ExperimentDesigner()
        
        # Platform state
        self.research_projects = {}
        self.models = {}
        self.anomalies = {}
        self.strategies = {}
        
        logger.info("Quantitative Trading Research Platform initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load platform configuration"""
        # Default configuration
        config = {
            "data_dir": "data",
            "models_dir": "models",
            "reports_dir": "reports",
            "research_dir": "research",
            "default_markets": ["SPY", "QQQ", "IWM", "GLD", "TLT"],
            "default_timeframe": "1d",
            "default_lookback": 252,
            "risk_free_rate": 0.02,
            "transaction_costs": 0.001,
            "slippage": 0.0005,
            "max_positions": 10,
            "max_drawdown": 0.20,
            "target_sharpe": 1.5,
            "research_methodology": "systematic"
        }
        
        # Load from file if provided
        if config_path and Path(config_path).exists():
            import json
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        
        return config
    
    def start_research_project(
        self,
        project_name: str,
        markets: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Start a new research project
        
        Args:
            project_name: Name of the research project
            markets: List of markets to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Research project configuration
        """
        logger.info(f"Starting research project: {project_name}")
        
        if markets is None:
            markets = self.config["default_markets"]
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        project_config = {
            "name": project_name,
            "markets": markets,
            "start_date": start_date,
            "end_date": end_date,
            "created_at": datetime.now(),
            "status": "active",
            "results": {}
        }
        
        self.research_projects[project_name] = project_config
        
        logger.info(f"Research project '{project_name}' created with {len(markets)} markets")
        return project_config
    
    def collect_market_data(
        self,
        project_name: str,
        markets: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Collect market data for research project
        
        Args:
            project_name: Name of the research project
            markets: List of markets to collect data for
            
        Returns:
            Market data DataFrame
        """
        project = self.research_projects.get(project_name)
        if not project:
            raise ValueError(f"Research project '{project_name}' not found")
        
        if markets is None:
            markets = project["markets"]
        
        logger.info(f"Collecting market data for {len(markets)} markets")
        
        # Collect data for each market
        all_data = []
        for market in markets:
            try:
                market_data = self.data_collector.collect_data(
                    symbol=market,
                    start_date=project["start_date"],
                    end_date=project["end_date"],
                    timeframe=self.config["default_timeframe"]
                )
                market_data['symbol'] = market
                all_data.append(market_data)
                logger.info(f"Collected data for {market}: {len(market_data)} records")
            except Exception as e:
                logger.warning(f"Failed to collect data for {market}: {e}")
        
        if not all_data:
            raise ValueError("No market data collected")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values(['symbol', 'timestamp'])
        
        # Store in project
        project["results"]["market_data"] = combined_data
        
        logger.info(f"Collected {len(combined_data)} total records")
        return combined_data
    
    def detect_anomalies(
        self,
        project_name: str,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Detect market anomalies for research project
        
        Args:
            project_name: Name of the research project
            market_data: Market data (if None, uses project data)
            
        Returns:
            Anomaly analysis results
        """
        project = self.research_projects.get(project_name)
        if not project:
            raise ValueError(f"Research project '{project_name}' not found")
        
        if market_data is None:
            market_data = project["results"].get("market_data")
            if market_data is None:
                raise ValueError("No market data available. Run collect_market_data first.")
        
        logger.info(f"Detecting anomalies for project: {project_name}")
        
        # Detect anomalies
        anomaly_analysis = self.anomaly_detector.find_anomalies(market_data)
        
        # Store results
        project["results"]["anomalies"] = anomaly_analysis
        
        logger.info(f"Detected {len(anomaly_analysis.anomalies)} anomalies")
        logger.info(f"Anomaly types: {[a.anomaly_type.value for a in anomaly_analysis.anomalies[:5]]}")
        
        return {
            "total_anomalies": len(anomaly_analysis.anomalies),
            "anomaly_types": list(set([a.anomaly_type.value for a in anomaly_analysis.anomalies])),
            "summary_stats": anomaly_analysis.summary_stats,
            "recommendations": anomaly_analysis.recommendations
        }
    
    def engineer_features(
        self,
        project_name: str,
        market_data: Optional[pd.DataFrame] = None,
        anomaly_data: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Engineer features for machine learning models
        
        Args:
            project_name: Name of the research project
            market_data: Market data (if None, uses project data)
            anomaly_data: Anomaly data (if None, uses project data)
            
        Returns:
            Feature matrix
        """
        project = self.research_projects.get(project_name)
        if not project:
            raise ValueError(f"Research project '{project_name}' not found")
        
        if market_data is None:
            market_data = project["results"].get("market_data")
            if market_data is None:
                raise ValueError("No market data available. Run collect_market_data first.")
        
        if anomaly_data is None:
            anomaly_data = project["results"].get("anomalies")
        
        logger.info(f"Engineering features for project: {project_name}")
        
        # Engineer features
        features = self.feature_engineer.create_features(
            market_data=market_data,
            anomaly_data=anomaly_data,
            lookback_period=self.config["default_lookback"]
        )
        
        # Store results
        project["results"]["features"] = features
        
        logger.info(f"Engineered {features.shape[1]} features for {features.shape[0]} samples")
        return features
    
    def build_models(
        self,
        project_name: str,
        features: Optional[pd.DataFrame] = None,
        target: Optional[pd.Series] = None,
        model_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Build machine learning models
        
        Args:
            project_name: Name of the research project
            features: Feature matrix (if None, uses project data)
            target: Target variable (if None, creates returns target)
            model_types: List of model types to build
            
        Returns:
            Model building results
        """
        project = self.research_projects.get(project_name)
        if not project:
            raise ValueError(f"Research project '{project_name}' not found")
        
        if features is None:
            features = project["results"].get("features")
            if features is None:
                raise ValueError("No features available. Run engineer_features first.")
        
        if target is None:
            # Create target variable (next period returns)
            market_data = project["results"]["market_data"]
            target = self._create_target_variable(market_data)
        
        if model_types is None:
            model_types = ["random_forest", "xgboost", "neural_net"]
        
        logger.info(f"Building models for project: {project_name}")
        
        # Build individual models
        models = {}
        for model_type in model_types:
            try:
                model = self.model_factory.create_model(
                    model_type=ModelType(model_type),
                    features=features,
                    target=target,
                    name=f"{project_name}_{model_type}"
                )
                models[model_type] = model
                logger.info(f"Built {model_type} model: {model.performance.test_score:.3f} test score")
            except Exception as e:
                logger.warning(f"Failed to build {model_type} model: {e}")
        
        # Build ensemble model
        if len(models) > 1:
            try:
                ensemble_model = self.model_factory.create_ensemble_model(
                    base_models=[ModelType(mt) for mt in models.keys()],
                    features=features,
                    target=target,
                    ensemble_method="voting",
                    name=f"{project_name}_ensemble"
                )
                models["ensemble"] = ensemble_model
                logger.info(f"Built ensemble model: {ensemble_model.performance.test_score:.3f} test score")
            except Exception as e:
                logger.warning(f"Failed to build ensemble model: {e}")
        
        # Store results
        project["results"]["models"] = models
        
        # Compare models
        model_names = [f"{project_name}_{mt}" for mt in models.keys()]
        comparison = self.model_factory.compare_models(model_names)
        
        logger.info(f"Built {len(models)} models successfully")
        return {
            "models_built": len(models),
            "model_types": list(models.keys()),
            "best_model": comparison.loc[comparison['Test Score'].idxmax(), 'Model'],
            "best_score": comparison['Test Score'].max(),
            "comparison": comparison
        }
    
    def _create_target_variable(self, market_data: pd.DataFrame) -> pd.Series:
        """Create target variable for prediction"""
        # Calculate returns for the first market (or average across markets)
        symbols = market_data['symbol'].unique()
        
        if len(symbols) == 1:
            # Single market
            symbol_data = market_data[market_data['symbol'] == symbols[0]].copy()
            target = symbol_data['close'].pct_change().shift(-1)  # Next period returns
        else:
            # Multiple markets - use average returns
            all_returns = []
            for symbol in symbols:
                symbol_data = market_data[market_data['symbol'] == symbol].copy()
                returns = symbol_data['close'].pct_change().shift(-1)
                all_returns.append(returns)
            
            # Average across markets
            target = pd.concat(all_returns, axis=1).mean(axis=1)
        
        # Remove NaN values
        target = target.dropna()
        
        return target
    
    def backtest_strategies(
        self,
        project_name: str,
        strategies: Optional[List[str]] = None,
        models: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Backtest trading strategies
        
        Args:
            project_name: Name of the research project
            strategies: List of strategies to test
            models: Models to use for strategy signals
            
        Returns:
            Backtesting results
        """
        project = self.research_projects.get(project_name)
        if not project:
            raise ValueError(f"Research project '{project_name}' not found")
        
        if strategies is None:
            strategies = ["mean_reversion", "momentum"]
        
        if models is None:
            models = project["results"].get("models", {})
        
        market_data = project["results"]["market_data"]
        
        logger.info(f"Backtesting strategies for project: {project_name}")
        
        # Backtest each strategy
        strategy_results = {}
        for strategy_name in strategies:
            try:
                if strategy_name == "mean_reversion":
                    strategy = MeanReversionStrategy()
                elif strategy_name == "momentum":
                    strategy = MomentumStrategy()
                else:
                    logger.warning(f"Unknown strategy: {strategy_name}")
                    continue
                
                # Backtest strategy
                results = self.backtester.backtest_strategy(
                    strategy=strategy,
                    data=market_data,
                    transaction_costs=self.config["transaction_costs"],
                    slippage=self.config["slippage"]
                )
                
                strategy_results[strategy_name] = results
                logger.info(f"Backtested {strategy_name}: {results['sharpe_ratio']:.3f} Sharpe ratio")
                
            except Exception as e:
                logger.warning(f"Failed to backtest {strategy_name}: {e}")
        
        # Store results
        project["results"]["strategy_results"] = strategy_results
        
        logger.info(f"Backtested {len(strategy_results)} strategies")
        return strategy_results
    
    def optimize_portfolio(
        self,
        project_name: str,
        strategies: Optional[Dict] = None,
        constraints: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Optimize portfolio allocation
        
        Args:
            project_name: Name of the research project
            strategies: Strategy results to optimize
            constraints: Portfolio constraints
            
        Returns:
            Portfolio optimization results
        """
        project = self.research_projects.get(project_name)
        if not project:
            raise ValueError(f"Research project '{project_name}' not found")
        
        if strategies is None:
            strategies = project["results"].get("strategy_results", {})
        
        if constraints is None:
            constraints = {
                "max_positions": self.config["max_positions"],
                "max_drawdown": self.config["max_drawdown"],
                "target_sharpe": self.config["target_sharpe"]
            }
        
        logger.info(f"Optimizing portfolio for project: {project_name}")
        
        # Optimize portfolio
        optimization_results = self.portfolio_optimizer.optimize_portfolio(
            strategies=strategies,
            constraints=constraints,
            risk_free_rate=self.config["risk_free_rate"]
        )
        
        # Store results
        project["results"]["portfolio_optimization"] = optimization_results
        
        logger.info(f"Portfolio optimization completed")
        logger.info(f"Optimal Sharpe ratio: {optimization_results.get('optimal_sharpe', 0):.3f}")
        
        return optimization_results
    
    def generate_hypotheses(
        self,
        project_name: str,
        anomaly_data: Optional[Any] = None,
        market_data: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate trading hypotheses
        
        Args:
            project_name: Name of the research project
            anomaly_data: Anomaly analysis data
            market_data: Market data
            
        Returns:
            List of hypotheses
        """
        project = self.research_projects.get(project_name)
        if not project:
            raise ValueError(f"Research project '{project_name}' not found")
        
        if anomaly_data is None:
            anomaly_data = project["results"].get("anomalies")
        
        if market_data is None:
            market_data = project["results"].get("market_data")
        
        logger.info(f"Generating hypotheses for project: {project_name}")
        
        # Generate hypotheses
        hypotheses = self.hypothesis_generator.generate_hypotheses(
            anomaly_data=anomaly_data,
            market_data=market_data,
            config=self.config
        )
        
        # Store results
        project["results"]["hypotheses"] = hypotheses
        
        logger.info(f"Generated {len(hypotheses)} hypotheses")
        return hypotheses
    
    def design_experiments(
        self,
        project_name: str,
        hypotheses: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Design experiments to test hypotheses
        
        Args:
            project_name: Name of the research project
            hypotheses: List of hypotheses to test
            
        Returns:
            List of experiments
        """
        project = self.research_projects.get(project_name)
        if not project:
            raise ValueError(f"Research project '{project_name}' not found")
        
        if hypotheses is None:
            hypotheses = project["results"].get("hypotheses", [])
        
        logger.info(f"Designing experiments for project: {project_name}")
        
        # Design experiments
        experiments = self.experiment_designer.design_experiments(
            hypotheses=hypotheses,
            config=self.config
        )
        
        # Store results
        project["results"]["experiments"] = experiments
        
        logger.info(f"Designed {len(experiments)} experiments")
        return experiments
    
    def run_complete_research_workflow(
        self,
        project_name: str,
        markets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run complete research workflow
        
        Args:
            project_name: Name of the research project
            markets: List of markets to analyze
            
        Returns:
            Complete research results
        """
        logger.info(f"Running complete research workflow for: {project_name}")
        
        # 1. Start research project
        project_config = self.start_research_project(project_name, markets)
        
        # 2. Collect market data
        market_data = self.collect_market_data(project_name)
        
        # 3. Detect anomalies
        anomaly_results = self.detect_anomalies(project_name, market_data)
        
        # 4. Engineer features
        features = self.engineer_features(project_name, market_data)
        
        # 5. Build models
        model_results = self.build_models(project_name, features)
        
        # 6. Backtest strategies
        strategy_results = self.backtest_strategies(project_name)
        
        # 7. Optimize portfolio
        portfolio_results = self.optimize_portfolio(project_name, strategy_results)
        
        # 8. Generate hypotheses
        hypotheses = self.generate_hypotheses(project_name)
        
        # 9. Design experiments
        experiments = self.design_experiments(project_name, hypotheses)
        
        # Compile results
        complete_results = {
            "project_config": project_config,
            "market_data_summary": {
                "total_records": len(market_data),
                "markets": list(market_data['symbol'].unique()),
                "date_range": f"{market_data['timestamp'].min()} to {market_data['timestamp'].max()}"
            },
            "anomaly_results": anomaly_results,
            "model_results": model_results,
            "strategy_results": strategy_results,
            "portfolio_results": portfolio_results,
            "hypotheses": hypotheses,
            "experiments": experiments,
            "recommendations": self._generate_workflow_recommendations(
                anomaly_results, model_results, strategy_results, portfolio_results
            )
        }
        
        logger.info(f"Complete research workflow finished for: {project_name}")
        return complete_results
    
    def _generate_workflow_recommendations(
        self,
        anomaly_results: Dict,
        model_results: Dict,
        strategy_results: Dict,
        portfolio_results: Dict
    ) -> List[str]:
        """Generate recommendations based on workflow results"""
        recommendations = []
        
        # Anomaly-based recommendations
        if anomaly_results.get("total_anomalies", 0) > 0:
            recommendations.append(f"Found {anomaly_results['total_anomalies']} anomalies - focus on persistent types")
            if "price_anomaly" in anomaly_results.get("anomaly_types", []):
                recommendations.append("Price anomalies detected - consider mean reversion strategies")
            if "volume_anomaly" in anomaly_results.get("anomaly_types", []):
                recommendations.append("Volume anomalies detected - consider momentum strategies")
        
        # Model-based recommendations
        if model_results.get("best_score", 0) > 0.1:
            recommendations.append(f"Best model achieved {model_results['best_score']:.3f} test score - good predictive power")
        else:
            recommendations.append("Low model performance - consider feature engineering or different models")
        
        # Strategy-based recommendations
        if strategy_results:
            best_strategy = max(strategy_results.keys(), 
                              key=lambda x: strategy_results[x].get('sharpe_ratio', 0))
            best_sharpe = strategy_results[best_strategy].get('sharpe_ratio', 0)
            recommendations.append(f"Best strategy: {best_strategy} (Sharpe: {best_sharpe:.3f})")
        
        # Portfolio-based recommendations
        if portfolio_results.get("optimal_sharpe", 0) > 1.0:
            recommendations.append("Portfolio optimization successful - consider live trading")
        else:
            recommendations.append("Portfolio needs improvement - review strategy selection")
        
        # General recommendations
        recommendations.append("Continue systematic testing and iteration")
        recommendations.append("Monitor model performance and retrain as needed")
        recommendations.append("Consider transaction costs and market impact in live trading")
        
        return recommendations
    
    def generate_research_report(
        self,
        project_name: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive research report
        
        Args:
            project_name: Name of the research project
            output_path: Output file path
            
        Returns:
            Report file path
        """
        project = self.research_projects.get(project_name)
        if not project:
            raise ValueError(f"Research project '{project_name}' not found")
        
        if output_path is None:
            output_path = f"reports/{project_name}_report.html"
        
        logger.info(f"Generating research report for: {project_name}")
        
        # Generate report content
        report_content = self._generate_report_content(project)
        
        # Ensure reports directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write report
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Research report saved to: {output_path}")
        return output_path
    
    def _generate_report_content(self, project: Dict) -> str:
        """Generate HTML report content"""
        results = project["results"]
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Research Report: {project['name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e8f4f8; border-radius: 3px; }}
                .recommendation {{ background: #fff3cd; padding: 10px; border-radius: 3px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Quantitative Trading Research Report</h1>
                <h2>Project: {project['name']}</h2>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Project Overview
        html += f'''
        <div class="section">
            <h2>Project Overview</h2>
            <div class="metric"><strong>Markets:</strong> {len(project['markets'])}</div>
            <div class="metric"><strong>Date Range:</strong> {project['start_date']} to {project['end_date']}</div>
            <div class="metric"><strong>Status:</strong> {project['status']}</div>
        </div>
        '''
        
        # Market Data Summary
        if "market_data" in results:
            market_data = results["market_data"]
            html += f'''
            <div class="section">
                <h2>Market Data</h2>
                <div class="metric"><strong>Total Records:</strong> {len(market_data):,}</div>
                <div class="metric"><strong>Markets:</strong> {', '.join(market_data['symbol'].unique())}</div>
                <div class="metric"><strong>Date Range:</strong> {market_data['timestamp'].min()} to {market_data['timestamp'].max()}</div>
            </div>
            '''
        
        # Anomaly Analysis
        if "anomalies" in results:
            anomalies = results["anomalies"]
            html += f'''
            <div class="section">
                <h2>Anomaly Analysis</h2>
                <div class="metric"><strong>Total Anomalies:</strong> {len(anomalies.anomalies)}</div>
                <div class="metric"><strong>Types:</strong> {', '.join(set([a.anomaly_type.value for a in anomalies.anomalies]))}</div>
            </div>
            '''
        
        # Model Results
        if "models" in results:
            models = results["models"]
            html += f'''
            <div class="section">
                <h2>Model Performance</h2>
                <div class="metric"><strong>Models Built:</strong> {len(models)}</div>
            '''
            
            for model_name, model in models.items():
                if model.performance:
                    html += f'''
                    <div class="metric">
                        <strong>{model_name}:</strong> {model.performance.test_score:.3f} test score, 
                        {model.performance.sharpe_ratio:.3f} Sharpe
                    </div>
                    '''
            
            html += '</div>'
        
        # Strategy Results
        if "strategy_results" in results:
            strategy_results = results["strategy_results"]
            html += f'''
            <div class="section">
                <h2>Strategy Performance</h2>
            '''
            
            for strategy_name, results in strategy_results.items():
                html += f'''
                <div class="metric">
                    <strong>{strategy_name}:</strong> {results.get('sharpe_ratio', 0):.3f} Sharpe ratio
                </div>
                '''
            
            html += '</div>'
        
        # Portfolio Optimization
        if "portfolio_optimization" in results:
            portfolio_results = results["portfolio_optimization"]
            html += f'''
            <div class="section">
                <h2>Portfolio Optimization</h2>
                <div class="metric"><strong>Optimal Sharpe:</strong> {portfolio_results.get('optimal_sharpe', 0):.3f}</div>
                <div class="metric"><strong>Max Drawdown:</strong> {portfolio_results.get('max_drawdown', 0):.1%}</div>
            </div>
            '''
        
        # Recommendations
        if "recommendations" in results:
            html += f'''
            <div class="section">
                <h2>Recommendations</h2>
            '''
            
            for rec in results["recommendations"]:
                html += f'<div class="recommendation">• {rec}</div>'
            
            html += '</div>'
        
        html += """
        </body>
        </html>
        """
        
        return html


def main():
    """Main entry point with example usage"""
    print("🧮 Quantitative Trading Research Platform")
    print("Inspired by Jim Simons and top algorithmic traders")
    print("Systematic research for finding many small edges\n")
    
    # Initialize platform
    platform = QuantitativeTradingResearchPlatform()
    
    try:
        # Run complete research workflow
        print("🚀 Running complete research workflow...")
        results = platform.run_complete_research_workflow(
            project_name="example_research",
            markets=["SPY", "QQQ", "IWM"]
        )
        
        # Display key results
        print("\n📊 Research Results Summary:")
        print(f"Project: {results['project_config']['name']}")
        print(f"Markets analyzed: {len(results['market_data_summary']['markets'])}")
        print(f"Total data records: {results['market_data_summary']['total_records']:,}")
        print(f"Anomalies detected: {results['anomaly_results']['total_anomalies']}")
        print(f"Models built: {results['model_results']['models_built']}")
        print(f"Best model score: {results['model_results']['best_score']:.3f}")
        print(f"Strategies tested: {len(results['strategy_results'])}")
        
        # Show recommendations
        print("\n💡 Key Recommendations:")
        for i, rec in enumerate(results['recommendations'][:5], 1):
            print(f"{i}. {rec}")
        
        # Generate report
        print("\n📋 Generating research report...")
        report_path = platform.generate_research_report("example_research")
        print(f"✅ Report saved to: {report_path}")
        
        print("\n🎯 Research Workflow Complete!")
        print("Key insights from Jim Simons approach:")
        print("• Focus on many small edges, not one magic formula")
        print("• Build complex, evolving machine learning models")
        print("• Test and iterate constantly")
        print("• Understand trading costs and market impact")
        print("• Collaborate and share knowledge")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 