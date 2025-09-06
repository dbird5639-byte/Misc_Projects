"""
MultiWalk-Inspired Testing Framework
Inspired by Kevin Davy's MultiWalk tool for automated strategy testing

This module provides comprehensive strategy testing across multiple markets,
proper out-of-sample validation, and performance filtering to evaluate
strategy robustness.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Results from strategy testing"""
    strategy_name: str
    market: str
    timeframe: str
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade: float
    avg_win: float
    avg_loss: float
    max_consecutive_losses: int
    calmar_ratio: float
    sortino_ratio: float
    equity_curve: pd.Series
    trades: List[Dict]
    in_sample_performance: Dict
    out_of_sample_performance: Dict
    walk_forward_results: List[Dict]
    market_regime_performance: Dict
    decay_analysis: Dict


class MultiWalkTester:
    """
    Kevin Davy's MultiWalk-inspired testing framework
    
    Tests strategies across multiple markets and timeframes with:
    - Proper out-of-sample validation
    - Walk-forward analysis
    - Market regime testing
    - Performance decay detection
    - Robustness evaluation
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the MultiWalk tester"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Default markets (Kevin's preferred futures)
        self.default_markets = {
            "ES": "E-mini S&P 500",
            "NQ": "E-mini NASDAQ-100", 
            "YM": "E-mini Dow Jones",
            "CL": "Crude Oil",
            "GC": "Gold",
            "ZB": "30-Year Treasury Bond",
            "ZC": "Corn",
            "ZS": "Soybeans",
            "ZW": "Wheat",
            "6E": "Euro FX"
        }
        
        # Default timeframes
        self.default_timeframes = ["daily", "4h", "1h", "30m", "15m"]
        
        logger.info("MultiWalk-inspired tester initialized")
    
    def test_strategy(
        self,
        strategy: Any,
        markets: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        years: int = 10,
        walk_forward_periods: int = 12,
        min_trades: int = 30,
        max_drawdown_threshold: float = 0.25
    ) -> TestResult:
        """
        Test a strategy across multiple markets and timeframes
        
        Args:
            strategy: Strategy object to test
            markets: List of markets to test on
            timeframes: List of timeframes to test on
            years: Number of years of historical data
            walk_forward_periods: Number of walk-forward periods
            min_trades: Minimum number of trades required
            max_drawdown_threshold: Maximum acceptable drawdown
            
        Returns:
            Aggregated test results
        """
        if markets is None:
            markets = list(self.default_markets.keys())
        
        if timeframes is None:
            timeframes = self.default_timeframes
        
        logger.info(f"Testing strategy {strategy.name} across {len(markets)} markets and {len(timeframes)} timeframes")
        
        # Test across all markets and timeframes
        all_results = []
        for market in markets:
            for timeframe in timeframes:
                try:
                    result = self._test_single_market_timeframe(
                        strategy=strategy,
                        market=market,
                        timeframe=timeframe,
                        years=years,
                        walk_forward_periods=walk_forward_periods
                    )
                    
                    if result and result.total_trades >= min_trades:
                        all_results.append(result)
                        logger.info(f"✅ {market} {timeframe}: {result.total_return:.2%} return, {result.max_drawdown:.2%} drawdown")
                    else:
                        logger.warning(f"⚠️ {market} {timeframe}: Insufficient trades or failed test")
                        
                except Exception as e:
                    logger.error(f"❌ Error testing {market} {timeframe}: {e}")
        
        if not all_results:
            raise ValueError("No valid test results obtained")
        
        # Aggregate results
        aggregated_result = self._aggregate_results(all_results, strategy.name)
        
        # Apply Kevin's performance filters
        self._apply_performance_filters(aggregated_result)
        
        return aggregated_result
    
    def _test_single_market_timeframe(
        self,
        strategy: Any,
        market: str,
        timeframe: str,
        years: int,
        walk_forward_periods: int
    ) -> Optional[TestResult]:
        """Test strategy on a single market and timeframe"""
        
        # Load market data
        data = self._load_market_data(market, timeframe, years)
        if data is None or len(data) < 252 * years * 0.5:  # At least 50% of requested data
            return None
        
        # Perform walk-forward analysis
        walk_forward_results = self._walk_forward_analysis(
            strategy=strategy,
            data=data,
            periods=walk_forward_periods
        )
        
        # Run full backtest
        full_result = self._run_backtest(strategy, data)
        
        # Analyze market regimes
        regime_performance = self._analyze_market_regimes(strategy, data)
        
        # Analyze strategy decay
        decay_analysis = self._analyze_strategy_decay(walk_forward_results)
        
        # Combine results
        result = TestResult(
            strategy_name=strategy.name,
            market=market,
            timeframe=timeframe,
            total_return=full_result["total_return"],
            max_drawdown=full_result["max_drawdown"],
            sharpe_ratio=full_result["sharpe_ratio"],
            win_rate=full_result["win_rate"],
            profit_factor=full_result["profit_factor"],
            total_trades=full_result["total_trades"],
            avg_trade=full_result["avg_trade"],
            avg_win=full_result["avg_win"],
            avg_loss=full_result["avg_loss"],
            max_consecutive_losses=full_result["max_consecutive_losses"],
            calmar_ratio=full_result["calmar_ratio"],
            sortino_ratio=full_result["sortino_ratio"],
            equity_curve=full_result["equity_curve"],
            trades=full_result["trades"],
            in_sample_performance=walk_forward_results["in_sample"],
            out_of_sample_performance=walk_forward_results["out_of_sample"],
            walk_forward_results=walk_forward_results["periods"],
            market_regime_performance=regime_performance,
            decay_analysis=decay_analysis
        )
        
        return result
    
    def _load_market_data(self, market: str, timeframe: str, years: int) -> Optional[pd.DataFrame]:
        """Load market data for testing"""
        try:
            # Try to load from local storage first
            data_file = self.data_dir / f"{market}_{timeframe}_{years}y.csv"
            if data_file.exists():
                data = pd.read_csv(data_file, index_col=0, parse_dates=True)
                logger.info(f"Loaded {market} {timeframe} data from cache")
                return data
            
            # Load from external source (simulated for demo)
            data = self._simulate_market_data(market, timeframe, years)
            
            # Save to cache
            data.to_csv(data_file)
            logger.info(f"Downloaded and cached {market} {timeframe} data")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data for {market} {timeframe}: {e}")
            return None
    
    def _simulate_market_data(self, market: str, timeframe: str, years: int) -> pd.DataFrame:
        """Simulate market data for demonstration"""
        # Generate realistic market data
        np.random.seed(hash(f"{market}_{timeframe}") % 2**32)
        
        # Calculate number of periods
        if timeframe == "daily":
            periods = years * 252
        elif timeframe == "4h":
            periods = years * 252 * 6
        elif timeframe == "1h":
            periods = years * 252 * 24
        else:
            periods = years * 252 * 24 * 4
        
        # Generate price data with realistic characteristics
        returns = np.random.normal(0.0001, 0.02, periods)  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Add some trend and volatility clustering
        trend = np.linspace(0, 0.1, periods)
        volatility = 0.02 + 0.01 * np.sin(np.linspace(0, 4*np.pi, periods))
        
        prices = prices * (1 + trend) * (1 + np.random.normal(0, volatility))
        
        # Create OHLC data
        dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='D')
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, periods)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, periods)
        }, index=dates)
        
        return data
    
    def _walk_forward_analysis(
        self,
        strategy: Any,
        data: pd.DataFrame,
        periods: int
    ) -> Dict:
        """
        Perform walk-forward analysis to prevent overfitting
        
        This is a key component of Kevin's MultiWalk approach
        """
        logger.info(f"Performing walk-forward analysis with {periods} periods")
        
        # Split data into periods
        total_length = len(data)
        period_length = total_length // periods
        
        in_sample_results = []
        out_of_sample_results = []
        period_results = []
        
        for i in range(periods - 1):
            # Define in-sample and out-of-sample periods
            in_sample_end = (i + 1) * period_length
            out_of_sample_end = (i + 2) * period_length
            
            in_sample_data = data.iloc[:in_sample_end]
            out_of_sample_data = data.iloc[in_sample_end:out_of_sample_end]
            
            # Test on in-sample data
            in_sample_result = self._run_backtest(strategy, in_sample_data)
            in_sample_results.append(in_sample_result)
            
            # Test on out-of-sample data
            out_of_sample_result = self._run_backtest(strategy, out_of_sample_data)
            out_of_sample_results.append(out_of_sample_result)
            
            # Store period results
            period_results.append({
                "period": i + 1,
                "in_sample": in_sample_result,
                "out_of_sample": out_of_sample_result,
                "performance_decay": out_of_sample_result["total_return"] - in_sample_result["total_return"]
            })
        
        # Aggregate results
        in_sample_avg = self._average_results(in_sample_results)
        out_of_sample_avg = self._average_results(out_of_sample_results)
        
        return {
            "in_sample": in_sample_avg,
            "out_of_sample": out_of_sample_avg,
            "periods": period_results,
            "avg_decay": np.mean([p["performance_decay"] for p in period_results])
        }
    
    def _run_backtest(self, strategy: Any, data: pd.DataFrame) -> Dict:
        """Run backtest on the strategy"""
        try:
            # Generate signals
            signals = strategy.generate_signals(data)
            
            # Calculate equity curve
            equity_curve = self._calculate_equity_curve(signals, data)
            
            # Extract trades
            trades = self._extract_trades(signals, data)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(equity_curve, trades)
            
            return {
                "total_return": metrics["total_return"],
                "max_drawdown": metrics["max_drawdown"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "total_trades": metrics["total_trades"],
                "avg_trade": metrics["avg_trade"],
                "avg_win": metrics["avg_win"],
                "avg_loss": metrics["avg_loss"],
                "max_consecutive_losses": metrics["max_consecutive_losses"],
                "calmar_ratio": metrics["calmar_ratio"],
                "sortino_ratio": metrics["sortino_ratio"],
                "equity_curve": equity_curve,
                "trades": trades
            }
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return self._empty_backtest_result()
    
    def _calculate_equity_curve(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
        """Calculate equity curve from signals"""
        # Simple equity curve calculation
        position = 0
        equity = [100]  # Start with $100
        
        for i in range(1, len(signals)):
            if signals.iloc[i]['signal'] == 1 and position <= 0:
                position = 1
            elif signals.iloc[i]['signal'] == -1 and position >= 0:
                position = -1
            elif signals.iloc[i]['signal'] == 0:
                position = 0
            
            # Calculate daily return
            daily_return = position * (data.iloc[i]['close'] - data.iloc[i-1]['close']) / data.iloc[i-1]['close']
            equity.append(equity[-1] * (1 + daily_return))
        
        return pd.Series(equity, index=data.index)
    
    def _extract_trades(self, signals: pd.DataFrame, data: pd.DataFrame) -> List[Dict]:
        """Extract individual trades from signals"""
        trades = []
        position = 0
        entry_price = 0
        entry_date = None
        
        for i, (date, row) in enumerate(signals.iterrows()):
            if row['signal'] == 1 and position <= 0:
                # Enter long position
                if position == -1:  # Close short position
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': data.iloc[i]['close'],
                        'position': 'short',
                        'pnl': entry_price - data.iloc[i]['close'],
                        'return': (entry_price - data.iloc[i]['close']) / entry_price
                    })
                
                position = 1
                entry_price = data.iloc[i]['close']
                entry_date = date
                
            elif row['signal'] == -1 and position >= 0:
                # Enter short position
                if position == 1:  # Close long position
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': data.iloc[i]['close'],
                        'position': 'long',
                        'pnl': data.iloc[i]['close'] - entry_price,
                        'return': (data.iloc[i]['close'] - entry_price) / entry_price
                    })
                
                position = -1
                entry_price = data.iloc[i]['close']
                entry_date = date
        
        return trades
    
    def _calculate_performance_metrics(self, equity_curve: pd.Series, trades: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return self._empty_metrics()
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        returns = equity_curve.pct_change().dropna()
        
        # Drawdown calculation
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Trade metrics
        pnls = [trade['pnl'] for trade in trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_trade = np.mean(pnls) if pnls else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        # Risk metrics
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        sortino_ratio = returns.mean() / returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for pnl in pnls:
            if pnl < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(trades),
            "avg_trade": avg_trade,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_consecutive_losses": max_consecutive_losses
        }
    
    def _analyze_market_regimes(self, strategy: Any, data: pd.DataFrame) -> Dict:
        """Analyze strategy performance across different market regimes"""
        # Simple market regime detection
        returns = data['close'].pct_change().dropna()
        
        # Volatility regime
        volatility = returns.rolling(20).std()
        high_vol = volatility > volatility.quantile(0.75)
        low_vol = volatility < volatility.quantile(0.25)
        
        # Trend regime
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        uptrend = sma_20 > sma_50
        downtrend = sma_20 < sma_50
        
        # Test strategy in different regimes
        regimes = {
            "high_volatility": high_vol,
            "low_volatility": low_vol,
            "uptrend": uptrend,
            "downtrend": downtrend
        }
        
        regime_performance = {}
        for regime_name, regime_mask in regimes.items():
            if regime_mask.sum() > 0:
                regime_data = data[regime_mask]
                if len(regime_data) > 50:  # Minimum data requirement
                    regime_result = self._run_backtest(strategy, regime_data)
                    regime_performance[regime_name] = {
                        "total_return": regime_result["total_return"],
                        "sharpe_ratio": regime_result["sharpe_ratio"],
                        "max_drawdown": regime_result["max_drawdown"],
                        "total_trades": regime_result["total_trades"]
                    }
        
        return regime_performance
    
    def _analyze_strategy_decay(self, walk_forward_results: Dict) -> Dict:
        """Analyze strategy performance decay over time"""
        periods = walk_forward_results["periods"]
        
        if not periods:
            return {"decay_rate": 0, "stability_score": 0}
        
        # Calculate decay rate
        in_sample_returns = [p["in_sample"]["total_return"] for p in periods]
        out_of_sample_returns = [p["out_of_sample"]["total_return"] for p in periods]
        
        decay_rates = [oos - ins for ins, oos in zip(in_sample_returns, out_of_sample_returns)]
        avg_decay_rate = np.mean(decay_rates)
        
        # Calculate stability score
        stability_score = 1 - abs(avg_decay_rate)  # Higher is better
        
        return {
            "decay_rate": avg_decay_rate,
            "stability_score": max(0, stability_score),
            "period_decay_rates": decay_rates,
            "in_sample_returns": in_sample_returns,
            "out_of_sample_returns": out_of_sample_returns
        }
    
    def _aggregate_results(self, results: List[TestResult], strategy_name: str) -> TestResult:
        """Aggregate results across markets and timeframes"""
        if not results:
            raise ValueError("No results to aggregate")
        
        # Calculate weighted averages
        total_trades = sum(r.total_trades for r in results)
        weights = [r.total_trades / total_trades for r in results]
        
        # Aggregate metrics
        total_return = sum(r.total_return * w for r, w in zip(results, weights))
        max_drawdown = max(r.max_drawdown for r in results)  # Worst case
        sharpe_ratio = sum(r.sharpe_ratio * w for r, w in zip(results, weights))
        win_rate = sum(r.win_rate * w for r, w in zip(results, weights))
        profit_factor = sum(r.profit_factor * w for r, w in zip(results, weights))
        
        # Combine equity curves
        combined_equity = pd.concat([r.equity_curve for r in results], axis=1).mean(axis=1)
        
        # Combine trades
        all_trades = []
        for r in results:
            for trade in r.trades:
                trade['market'] = r.market
                trade['timeframe'] = r.timeframe
                all_trades.append(trade)
        
        # Create aggregated result
        aggregated = TestResult(
            strategy_name=strategy_name,
            market="ALL",
            timeframe="ALL",
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade=sum(r.avg_trade * w for r, w in zip(results, weights)),
            avg_win=sum(r.avg_win * w for r, w in zip(results, weights)),
            avg_loss=sum(r.avg_loss * w for r, w in zip(results, weights)),
            max_consecutive_losses=max(r.max_consecutive_losses for r in results),
            calmar_ratio=total_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            sortino_ratio=sum(r.sortino_ratio * w for r, w in zip(results, weights)),
            equity_curve=combined_equity,
            trades=all_trades,
            in_sample_performance=self._average_results([r.in_sample_performance for r in results]),
            out_of_sample_performance=self._average_results([r.out_of_sample_performance for r in results]),
            walk_forward_results=results[0].walk_forward_results,  # Use first result as template
            market_regime_performance=self._combine_regime_performance([r.market_regime_performance for r in results]),
            decay_analysis=self._average_decay_analysis([r.decay_analysis for r in results])
        )
        
        return aggregated
    
    def _apply_performance_filters(self, result: TestResult) -> None:
        """Apply Kevin's performance filters"""
        # Kevin's 2x drawdown rule
        return_drawdown_ratio = result.total_return / abs(result.max_drawdown) if result.max_drawdown != 0 else 0
        
        if return_drawdown_ratio < 2.0:
            logger.warning(f"Strategy {result.strategy_name} fails Kevin's 2x drawdown rule: {return_drawdown_ratio:.2f}")
        
        # Minimum Sharpe ratio
        if result.sharpe_ratio < 0.5:
            logger.warning(f"Strategy {result.strategy_name} has low Sharpe ratio: {result.sharpe_ratio:.2f}")
        
        # Minimum win rate
        if result.win_rate < 0.4:
            logger.warning(f"Strategy {result.strategy_name} has low win rate: {result.win_rate:.2%}")
        
        # Maximum drawdown
        if result.max_drawdown < -0.25:
            logger.warning(f"Strategy {result.strategy_name} has high drawdown: {result.max_drawdown:.2%}")
    
    def _average_results(self, results: List[Dict]) -> Dict:
        """Average multiple result dictionaries"""
        if not results:
            return {}
        
        avg_result = {}
        for key in results[0].keys():
            if isinstance(results[0][key], (int, float)):
                avg_result[key] = np.mean([r[key] for r in results])
            else:
                avg_result[key] = results[0][key]  # Keep first non-numeric value
        
        return avg_result
    
    def _combine_regime_performance(self, regime_results: List[Dict]) -> Dict:
        """Combine market regime performance across markets"""
        combined = {}
        
        for regime_result in regime_results:
            for regime, performance in regime_result.items():
                if regime not in combined:
                    combined[regime] = []
                combined[regime].append(performance)
        
        # Average performance for each regime
        for regime in combined:
            combined[regime] = self._average_results(combined[regime])
        
        return combined
    
    def _average_decay_analysis(self, decay_results: List[Dict]) -> Dict:
        """Average decay analysis results"""
        if not decay_results:
            return {"decay_rate": 0, "stability_score": 0}
        
        return {
            "decay_rate": np.mean([d["decay_rate"] for d in decay_results]),
            "stability_score": np.mean([d["stability_score"] for d in decay_results])
        }
    
    def _empty_backtest_result(self) -> Dict:
        """Return empty backtest result"""
        return {
            "total_return": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "total_trades": 0,
            "avg_trade": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "max_consecutive_losses": 0,
            "calmar_ratio": 0,
            "sortino_ratio": 0,
            "equity_curve": pd.Series(),
            "trades": []
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics"""
        return {
            "total_return": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "calmar_ratio": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "total_trades": 0,
            "avg_trade": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "max_consecutive_losses": 0
        }


# Example usage
if __name__ == "__main__":
    # Create a simple test strategy
    class TestStrategy:
        def __init__(self):
            self.name = "test_strategy"
        
        def generate_signals(self, data):
            # Simple moving average crossover
            data = data.copy()
            data['sma_20'] = data['close'].rolling(20).mean()
            data['sma_50'] = data['close'].rolling(50).mean()
            
            data['signal'] = 0
            data.loc[data['sma_20'] > data['sma_50'], 'signal'] = 1
            data.loc[data['sma_20'] < data['sma_50'], 'signal'] = -1
            
            return data
    
    # Test the strategy
    tester = MultiWalkTester()
    strategy = TestStrategy()
    
    result = tester.test_strategy(
        strategy=strategy,
        markets=["ES", "NQ"],
        timeframes=["daily"],
        years=5
    )
    
    print(f"Strategy: {result.strategy_name}")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Return/Drawdown: {result.total_return / abs(result.max_drawdown):.2f}")
    print(f"Total Trades: {result.total_trades}") 