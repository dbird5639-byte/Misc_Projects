"""
Backtest Agent for validating trading strategies using historical data.
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from .model_factory import ModelFactory
from config.settings import Settings, BacktestConfig
from backtesting.backtest_engine import BacktestEngine
from backtesting.performance import PerformanceMetrics
from data.market_data import MarketDataManager
from utils.logger import setup_logger


@dataclass
class BacktestResult:
    """Results from strategy backtesting."""
    strategy_name: str
    is_profitable: bool
    meets_criteria: bool
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    optimal_parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    timestamp: datetime


class BacktestAgent:
    """
    AI-powered backtesting agent that validates trading strategies.
    """
    
    def __init__(self, model_factory: ModelFactory, settings: Settings):
        self.model_factory = model_factory
        self.settings = settings
        self.logger = setup_logger("backtest_agent", settings.log_level)
        
        # Initialize components
        self.backtest_engine = BacktestEngine(settings)
        self.market_data = MarketDataManager(settings)
        self.performance_calculator = PerformanceMetrics()
        
        # Backtesting criteria
        self.criteria = {
            "min_sharpe": settings.backtest_config.min_sharpe,
            "max_drawdown": settings.backtest_config.max_drawdown,
            "min_trades": settings.backtest_config.min_trades,
            "min_win_rate": 0.45,
            "min_profit_factor": 1.2
        }
        
        self.logger.info("Backtest Agent initialized")
    
    async def backtest_strategy(self, strategy) -> BacktestResult:
        """Backtest a trading strategy."""
        self.logger.info(f"Backtesting strategy: {strategy.name}")
        
        try:
            # Get historical data
            data = await self._get_historical_data(strategy)
            
            # Run initial backtest
            initial_result = await self._run_backtest(strategy, data)
            
            # Optimize parameters if needed
            if initial_result.meets_criteria:
                optimized_result = await self._optimize_parameters(strategy, data, initial_result)
                return optimized_result
            else:
                return initial_result
                
        except Exception as e:
            self.logger.error(f"Backtesting error for {strategy.name}: {e}")
            return self._create_failed_result(strategy)
    
    async def _get_historical_data(self, strategy) -> pd.DataFrame:
        """Get historical data for strategy backtesting."""
        # Get symbols from strategy or use defaults
        symbols = getattr(strategy, 'symbols', self.settings.data_config.symbols[:3])
        timeframe = getattr(strategy, 'timeframe', '1d')
        
        # Get data for each symbol
        all_data = {}
        for symbol in symbols:
            try:
                data = await self.market_data.get_historical_data(
                    symbol=symbol,
                    start_date=self.settings.backtest_config.start_date,
                    end_date=self.settings.backtest_config.end_date,
                    timeframe=timeframe
                )
                all_data[symbol] = data
            except Exception as e:
                self.logger.warning(f"Could not get data for {symbol}: {e}")
        
        # Combine data if multiple symbols
        if len(all_data) == 1:
            return list(all_data.values())[0]
        else:
            return self._combine_multi_symbol_data(all_data)
    
    def _combine_multi_symbol_data(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine data from multiple symbols."""
        # For simplicity, use the first symbol's data
        # In a real implementation, you'd combine them based on strategy needs
        return list(all_data.values())[0]
    
    async def _run_backtest(self, strategy, data: pd.DataFrame) -> BacktestResult:
        """Run a single backtest."""
        try:
            # Execute strategy on historical data
            results = await self.backtest_engine.run_backtest(
                strategy=strategy,
                data=data,
                initial_capital=self.settings.backtest_config.initial_capital,
                commission=self.settings.backtest_config.commission,
                slippage=self.settings.backtest_config.slippage
            )
            
            # Calculate performance metrics
            metrics = self.performance_calculator.calculate_metrics(results)
            
            # Evaluate against criteria
            meets_criteria = self._evaluate_criteria(metrics)
            
            return BacktestResult(
                strategy_name=strategy.name,
                is_profitable=metrics['total_return'] > 0,
                meets_criteria=meets_criteria,
                total_return=metrics['total_return'],
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                win_rate=metrics['win_rate'],
                profit_factor=metrics['profit_factor'],
                total_trades=metrics['total_trades'],
                avg_trade_duration=metrics['avg_trade_duration'],
                optimal_parameters=strategy.parameters,
                performance_metrics=metrics,
                equity_curve=results['equity_curve'],
                trade_log=results['trade_log'],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Backtest execution error: {e}")
            return self._create_failed_result(strategy)
    
    def _evaluate_criteria(self, metrics: Dict[str, float]) -> bool:
        """Evaluate if strategy meets backtesting criteria."""
        return (
            metrics['sharpe_ratio'] >= self.criteria['min_sharpe'] and
            metrics['max_drawdown'] <= self.criteria['max_drawdown'] and
            metrics['total_trades'] >= self.criteria['min_trades'] and
            metrics['win_rate'] >= self.criteria['min_win_rate'] and
            metrics['profit_factor'] >= self.criteria['min_profit_factor']
        )
    
    async def _optimize_parameters(self, strategy, data: pd.DataFrame, initial_result: BacktestResult) -> BacktestResult:
        """Optimize strategy parameters using AI."""
        self.logger.info(f"Optimizing parameters for {strategy.name}")
        
        try:
            # Use AI to suggest parameter optimizations
            optimization_prompt = self._create_optimization_prompt(strategy, initial_result)
            
            model = self.model_factory.get_model(self.settings.default_model)
            response = await model.generate(optimization_prompt)
            
            # Parse optimization suggestions
            optimized_params = self._parse_optimization_response(response, strategy.parameters)
            
            if optimized_params:
                # Update strategy with optimized parameters
                strategy.parameters.update(optimized_params)
                
                # Re-run backtest with optimized parameters
                optimized_result = await self._run_backtest(strategy, data)
                
                # Use the better result
                if optimized_result.sharpe_ratio > initial_result.sharpe_ratio:
                    return optimized_result
            
            return initial_result
            
        except Exception as e:
            self.logger.error(f"Parameter optimization error: {e}")
            return initial_result
    
    def _create_optimization_prompt(self, strategy, result: BacktestResult) -> str:
        """Create prompt for parameter optimization."""
        return f"""
You are an expert quantitative trader optimizing a trading strategy.

Strategy: {strategy.name}
Current Parameters: {strategy.parameters}
Current Performance:
- Sharpe Ratio: {result.sharpe_ratio:.3f}
- Total Return: {result.total_return:.2%}
- Max Drawdown: {result.max_drawdown:.2%}
- Win Rate: {result.win_rate:.2%}
- Profit Factor: {result.profit_factor:.3f}
- Total Trades: {result.total_trades}

Analyze the current performance and suggest parameter optimizations to improve:
1. Sharpe ratio
2. Risk-adjusted returns
3. Win rate
4. Profit factor
5. Maximum drawdown

Consider:
- Parameter sensitivity analysis
- Overfitting prevention
- Market regime adaptation
- Risk management improvements

Return your suggestions as a JSON object with:
- optimized_parameters: dict of suggested parameter changes
- reasoning: explanation for each change
- expected_improvement: expected performance improvement
- confidence: confidence in suggestions (0-1)

Focus on practical, implementable changes that improve risk-adjusted returns.
"""
    
    def _parse_optimization_response(self, response: str, current_params: Dict) -> Optional[Dict]:
        """Parse AI response to extract optimization suggestions."""
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('optimized_parameters', {})
        except Exception as e:
            self.logger.error(f"Error parsing optimization response: {e}")
        
        return None
    
    def _create_failed_result(self, strategy) -> BacktestResult:
        """Create a failed backtest result."""
        return BacktestResult(
            strategy_name=strategy.name,
            is_profitable=False,
            meets_criteria=False,
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            avg_trade_duration=0.0,
            optimal_parameters={},
            performance_metrics={},
            equity_curve=pd.Series(),
            trade_log=pd.DataFrame(),
            timestamp=datetime.now()
        )
    
    async def run_walk_forward_analysis(self, strategy, data: pd.DataFrame) -> Dict[str, Any]:
        """Run walk-forward analysis to prevent overfitting."""
        self.logger.info(f"Running walk-forward analysis for {strategy.name}")
        
        try:
            # Split data into training and testing periods
            periods = self._create_walk_forward_periods(data)
            
            results = []
            for i, (train_data, test_data) in enumerate(periods):
                # Train on training data
                train_result = await self._run_backtest(strategy, train_data)
                
                # Test on out-of-sample data
                test_result = await self._run_backtest(strategy, test_data)
                
                results.append({
                    'period': i,
                    'train_sharpe': train_result.sharpe_ratio,
                    'test_sharpe': test_result.sharpe_ratio,
                    'train_return': train_result.total_return,
                    'test_return': test_result.total_return,
                    'stability': test_result.sharpe_ratio / train_result.sharpe_ratio if train_result.sharpe_ratio > 0 else 0
                })
            
            # Calculate walk-forward metrics
            avg_train_sharpe = np.mean([r['train_sharpe'] for r in results])
            avg_test_sharpe = np.mean([r['test_sharpe'] for r in results])
            stability_score = avg_test_sharpe / avg_train_sharpe if avg_train_sharpe > 0 else 0
            
            return {
                'periods': results,
                'avg_train_sharpe': avg_train_sharpe,
                'avg_test_sharpe': avg_test_sharpe,
                'stability_score': stability_score,
                'is_stable': stability_score > 0.7  # 70% stability threshold
            }
            
        except Exception as e:
            self.logger.error(f"Walk-forward analysis error: {e}")
            return {'error': str(e)}
    
    def _create_walk_forward_periods(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create walk-forward analysis periods."""
        periods = []
        total_length = len(data)
        period_length = total_length // self.settings.backtest_config.walk_forward_periods
        
        for i in range(self.settings.backtest_config.walk_forward_periods - 1):
            train_end = (i + 1) * period_length
            test_end = (i + 2) * period_length
            
            train_data = data.iloc[:train_end]
            test_data = data.iloc[train_end:test_end]
            
            periods.append((train_data, test_data))
        
        return periods
    
    async def run_monte_carlo_simulation(self, strategy, data: pd.DataFrame, n_simulations: int = 1000) -> Dict[str, Any]:
        """Run Monte Carlo simulation for risk assessment."""
        self.logger.info(f"Running Monte Carlo simulation for {strategy.name}")
        
        try:
            # Get trade returns from backtest
            initial_result = await self._run_backtest(strategy, data)
            trade_returns = initial_result.trade_log['return'].dropna()
            
            if len(trade_returns) == 0:
                return {'error': 'No trades found for simulation'}
            
            # Run Monte Carlo simulations
            simulation_results = []
            for _ in range(n_simulations):
                # Bootstrap trade returns
                bootstrapped_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
                
                # Calculate cumulative returns
                cumulative_returns = np.cumprod(1 + bootstrapped_returns)
                
                # Calculate metrics
                total_return = cumulative_returns[-1] - 1
                max_drawdown = self._calculate_max_drawdown(cumulative_returns)
                sharpe_ratio = np.mean(bootstrapped_returns) / np.std(bootstrapped_returns) * np.sqrt(252) if np.std(bootstrapped_returns) > 0 else 0
                
                simulation_results.append({
                    'total_return': total_return,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio
                })
            
            # Calculate percentiles
            returns = [r['total_return'] for r in simulation_results]
            drawdowns = [r['max_drawdown'] for r in simulation_results]
            sharpes = [r['sharpe_ratio'] for r in simulation_results]
            
            return {
                'n_simulations': n_simulations,
                'percentiles': {
                    'returns': {
                        '5%': np.percentile(returns, 5),
                        '25%': np.percentile(returns, 25),
                        '50%': np.percentile(returns, 50),
                        '75%': np.percentile(returns, 75),
                        '95%': np.percentile(returns, 95)
                    },
                    'drawdowns': {
                        '5%': np.percentile(drawdowns, 5),
                        '25%': np.percentile(drawdowns, 25),
                        '50%': np.percentile(drawdowns, 50),
                        '75%': np.percentile(drawdowns, 75),
                        '95%': np.percentile(drawdowns, 95)
                    },
                    'sharpes': {
                        '5%': np.percentile(sharpes, 5),
                        '25%': np.percentile(sharpes, 25),
                        '50%': np.percentile(sharpes, 50),
                        '75%': np.percentile(sharpes, 75),
                        '95%': np.percentile(sharpes, 95)
                    }
                },
                'risk_metrics': {
                    'var_95': np.percentile(returns, 5),  # 95% VaR
                    'cvar_95': np.mean([r for r in returns if r <= np.percentile(returns, 5)]),  # 95% CVaR
                    'worst_case_drawdown': np.max(drawdowns)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Monte Carlo simulation error: {e}")
            return {'error': str(e)}
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns."""
        peak = cumulative_returns[0]
        max_dd = 0
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    async def generate_backtest_report(self, result: BacktestResult) -> str:
        """Generate a comprehensive backtest report."""
        try:
            prompt = f"""
Generate a comprehensive backtest report for this trading strategy:

Strategy: {result.strategy_name}
Performance Metrics:
- Total Return: {result.total_return:.2%}
- Sharpe Ratio: {result.sharpe_ratio:.3f}
- Max Drawdown: {result.max_drawdown:.2%}
- Win Rate: {result.win_rate:.2%}
- Profit Factor: {result.profit_factor:.3f}
- Total Trades: {result.total_trades}
- Avg Trade Duration: {result.avg_trade_duration:.1f} days

Parameters: {result.optimal_parameters}

Create a professional report including:
1. Executive Summary
2. Strategy Overview
3. Performance Analysis
4. Risk Assessment
5. Trade Analysis
6. Recommendations
7. Conclusion

Format as markdown with clear sections and insights.
"""
            
            model = self.model_factory.get_model(self.settings.default_model)
            response = await model.generate(prompt)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}" 