"""
Market Anomaly Detector
Detect subtle market anomalies and inefficiencies

Based on Jim Simons' approach of finding many small, subtle edges
rather than searching for a single overwhelming advantage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of market anomalies"""
    PRICE_ANOMALY = "price_anomaly"
    VOLUME_ANOMALY = "volume_anomaly"
    VOLATILITY_ANOMALY = "volatility_anomaly"
    CORRELATION_ANOMALY = "correlation_anomaly"
    MICROSTRUCTURE_ANOMALY = "microstructure_anomaly"
    REGIME_CHANGE = "regime_change"
    LIQUIDITY_ANOMALY = "liquidity_anomaly"
    MOMENTUM_ANOMALY = "momentum_anomaly"


@dataclass
class Anomaly:
    """Market anomaly information"""
    timestamp: datetime
    symbol: str
    anomaly_type: AnomalyType
    severity: float  # 0-1 scale
    confidence: float  # 0-1 scale
    description: str
    features: Dict[str, float]
    expected_return: Optional[float] = None
    decay_rate: Optional[float] = None


@dataclass
class AnomalyAnalysis:
    """Comprehensive anomaly analysis results"""
    anomalies: List[Anomaly]
    summary_stats: Dict[str, float]
    persistence_analysis: Dict[str, float]
    correlation_matrix: pd.DataFrame
    regime_changes: List[Dict[str, Any]]
    recommendations: List[str]


class MarketAnomalyDetector:
    """
    Market anomaly detector implementing Jim Simons' approach
    
    Focuses on finding subtle, persistent anomalies rather than
    obvious market inefficiencies that get arbitraged away quickly.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the anomaly detector"""
        self.config = config or self._default_config()
        self.detectors = self._initialize_detectors()
        
        logger.info("Market Anomaly Detector initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "lookback_period": 252,  # 1 year
            "min_anomaly_severity": 0.3,
            "min_confidence": 0.6,
            "persistence_threshold": 0.7,
            "correlation_threshold": 0.8,
            "regime_change_threshold": 0.05,
            "volume_threshold": 2.0,  # 2x average volume
            "volatility_threshold": 2.0,  # 2x average volatility
            "momentum_threshold": 0.02,  # 2% daily return
            "microstructure_window": 20,  # 20-minute windows
            "liquidity_threshold": 0.5  # 50% of average liquidity
        }
    
    def _initialize_detectors(self) -> Dict[str, Any]:
        """Initialize different anomaly detection algorithms"""
        return {
            "isolation_forest": IsolationForest(
                contamination=0.1,
                random_state=42
            ),
            "dbscan": DBSCAN(
                eps=0.5,
                min_samples=5
            ),
            "statistical": {
                "z_score_threshold": 3.0,
                "iqr_multiplier": 1.5
            }
        }
    
    def find_anomalies(
        self,
        market_data: pd.DataFrame,
        symbols: Optional[List[str]] = None
    ) -> AnomalyAnalysis:
        """
        Find market anomalies across multiple dimensions
        
        Args:
            market_data: DataFrame with OHLCV data
            symbols: List of symbols to analyze (None for all)
            
        Returns:
            Comprehensive anomaly analysis
        """
        logger.info("Starting comprehensive anomaly detection")
        
        if symbols is None:
            symbols = market_data['symbol'].unique().tolist()
        
        all_anomalies = []
        
        for symbol in symbols:
            symbol_data = market_data[market_data['symbol'] == symbol].copy()
            if len(symbol_data) < self.config["lookback_period"]:
                continue
            
            # Detect different types of anomalies
            price_anomalies = self._detect_price_anomalies(symbol_data, symbol)
            volume_anomalies = self._detect_volume_anomalies(symbol_data, symbol)
            volatility_anomalies = self._detect_volatility_anomalies(symbol_data, symbol)
            microstructure_anomalies = self._detect_microstructure_anomalies(symbol_data, symbol)
            momentum_anomalies = self._detect_momentum_anomalies(symbol_data, symbol)
            
            all_anomalies.extend(price_anomalies)
            all_anomalies.extend(volume_anomalies)
            all_anomalies.extend(volatility_anomalies)
            all_anomalies.extend(microstructure_anomalies)
            all_anomalies.extend(momentum_anomalies)
        
        # Detect cross-asset anomalies
        correlation_anomalies = self._detect_correlation_anomalies(market_data)
        all_anomalies.extend(correlation_anomalies)
        
        # Detect regime changes
        regime_changes = self._detect_regime_changes(market_data)
        
        # Filter and rank anomalies
        filtered_anomalies = self._filter_anomalies(all_anomalies)
        
        # Analyze persistence and correlations
        persistence_analysis = self._analyze_persistence(filtered_anomalies)
        correlation_matrix = self._calculate_anomaly_correlations(filtered_anomalies)
        
        # Generate summary statistics
        summary_stats = self._calculate_summary_stats(filtered_anomalies)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            filtered_anomalies, persistence_analysis, correlation_matrix
        )
        
        return AnomalyAnalysis(
            anomalies=filtered_anomalies,
            summary_stats=summary_stats,
            persistence_analysis=persistence_analysis,
            correlation_matrix=correlation_matrix,
            regime_changes=regime_changes,
            recommendations=recommendations
        )
    
    def _detect_price_anomalies(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> List[Anomaly]:
        """Detect price-based anomalies"""
        anomalies = []
        
        # Calculate returns
        data['returns'] = data['close'].pct_change()
        
        # Z-score based detection
        returns_mean = data['returns'].rolling(
            window=self.config["lookback_period"]
        ).mean()
        returns_std = data['returns'].rolling(
            window=self.config["lookback_period"]
        ).std()
        z_scores = (data['returns'] - returns_mean) / returns_std
        
        # Find extreme returns
        extreme_returns = z_scores.abs() > self.detectors["statistical"]["z_score_threshold"]
        
        for idx in data[extreme_returns].index:
            z_score = z_scores.loc[idx]
            severity = min(abs(z_score) / 5.0, 1.0)  # Normalize to 0-1
            confidence = min(abs(z_score) / 4.0, 1.0)  # Higher z-score = higher confidence
            
            if severity >= self.config["min_anomaly_severity"]:
                anomaly = Anomaly(
                    timestamp=data.loc[idx, 'timestamp'],
                    symbol=symbol,
                    anomaly_type=AnomalyType.PRICE_ANOMALY,
                    severity=severity,
                    confidence=confidence,
                    description=f"Extreme price movement: {z_score:.2f} z-score",
                    features={
                        'z_score': z_score,
                        'return': data.loc[idx, 'returns'],
                        'price': data.loc[idx, 'close']
                    }
                )
                anomalies.append(anomaly)
        
        # IQR based detection
        q1 = data['returns'].rolling(window=self.config["lookback_period"]).quantile(0.25)
        q3 = data['returns'].rolling(window=self.config["lookback_period"]).quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - self.detectors["statistical"]["iqr_multiplier"] * iqr
        upper_bound = q3 + self.detectors["statistical"]["iqr_multiplier"] * iqr
        
        iqr_outliers = (data['returns'] < lower_bound) | (data['returns'] > upper_bound)
        
        for idx in data[iqr_outliers].index:
            if idx not in [a.timestamp for a in anomalies]:  # Avoid duplicates
                return_val = data.loc[idx, 'returns']
                severity = min(abs(return_val) / 0.1, 1.0)  # Normalize to 0-1
                confidence = 0.7  # Moderate confidence for IQR method
                
                if severity >= self.config["min_anomaly_severity"]:
                    anomaly = Anomaly(
                        timestamp=data.loc[idx, 'timestamp'],
                        symbol=symbol,
                        anomaly_type=AnomalyType.PRICE_ANOMALY,
                        severity=severity,
                        confidence=confidence,
                        description=f"IQR outlier: {return_val:.3f} return",
                        features={
                            'return': return_val,
                            'iqr': iqr.loc[idx],
                            'price': data.loc[idx, 'close']
                        }
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_volume_anomalies(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> List[Anomaly]:
        """Detect volume-based anomalies"""
        anomalies = []
        
        # Calculate volume metrics
        data['volume_ma'] = data['volume'].rolling(
            window=self.config["lookback_period"]
        ).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Find volume spikes
        volume_spikes = data['volume_ratio'] > self.config["volume_threshold"]
        
        for idx in data[volume_spikes].index:
            volume_ratio = data.loc[idx, 'volume_ratio']
            severity = min((volume_ratio - 1) / 3.0, 1.0)  # Normalize to 0-1
            confidence = min(volume_ratio / 4.0, 1.0)
            
            if severity >= self.config["min_anomaly_severity"]:
                anomaly = Anomaly(
                    timestamp=data.loc[idx, 'timestamp'],
                    symbol=symbol,
                    anomaly_type=AnomalyType.VOLUME_ANOMALY,
                    severity=severity,
                    confidence=confidence,
                    description=f"Volume spike: {volume_ratio:.2f}x average",
                    features={
                        'volume_ratio': volume_ratio,
                        'volume': data.loc[idx, 'volume'],
                        'price': data.loc[idx, 'close']
                    }
                )
                anomalies.append(anomaly)
        
        # Detect volume-price divergence
        data['price_change'] = data['close'].pct_change()
        data['volume_change'] = data['volume'].pct_change()
        
        # Calculate correlation between price and volume changes
        correlation_window = 20
        correlations = data['price_change'].rolling(correlation_window).corr(data['volume_change'])
        
        # Find periods of negative correlation (divergence)
        divergence = correlations < -0.5
        for idx in data[divergence].index:
            if not pd.isna(correlations.loc[idx]):
                severity = min(abs(correlations.loc[idx]) / 0.8, 1.0)
                confidence = 0.6
                
                if severity >= self.config["min_anomaly_severity"]:
                    anomaly = Anomaly(
                        timestamp=data.loc[idx, 'timestamp'],
                        symbol=symbol,
                        anomaly_type=AnomalyType.VOLUME_ANOMALY,
                        severity=severity,
                        confidence=confidence,
                        description=f"Volume-price divergence: {correlations.loc[idx]:.3f} correlation",
                        features={
                            'correlation': correlations.loc[idx],
                            'price_change': data.loc[idx, 'price_change'],
                            'volume_change': data.loc[idx, 'volume_change']
                        }
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_volatility_anomalies(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> List[Anomaly]:
        """Detect volatility-based anomalies"""
        anomalies = []
        
        # Calculate realized volatility
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Calculate volatility of volatility
        data['vol_of_vol'] = data['volatility'].rolling(window=60).std()
        
        # Find volatility spikes
        vol_ma = data['volatility'].rolling(window=self.config["lookback_period"]).mean()
        vol_spikes = data['volatility'] > vol_ma * self.config["volatility_threshold"]
        
        for idx in data[vol_spikes].index:
            vol_ratio = data.loc[idx, 'volatility'] / vol_ma.loc[idx]
            severity = min((vol_ratio - 1) / 2.0, 1.0)
            confidence = min(vol_ratio / 3.0, 1.0)
            
            if severity >= self.config["min_anomaly_severity"]:
                anomaly = Anomaly(
                    timestamp=data.loc[idx, 'timestamp'],
                    symbol=symbol,
                    anomaly_type=AnomalyType.VOLATILITY_ANOMALY,
                    severity=severity,
                    confidence=confidence,
                    description=f"Volatility spike: {vol_ratio:.2f}x average",
                    features={
                        'volatility_ratio': vol_ratio,
                        'volatility': data.loc[idx, 'volatility'],
                        'vol_of_vol': data.loc[idx, 'vol_of_vol']
                    }
                )
                anomalies.append(anomaly)
        
        # Detect volatility regime changes
        vol_changes = data['volatility'].pct_change()
        regime_changes = vol_changes.abs() > 0.5  # 50% change in volatility
        
        for idx in data[regime_changes].index:
            if not pd.isna(vol_changes.loc[idx]):
                severity = min(abs(vol_changes.loc[idx]) / 1.0, 1.0)
                confidence = 0.7
                
                if severity >= self.config["min_anomaly_severity"]:
                    anomaly = Anomaly(
                        timestamp=data.loc[idx, 'timestamp'],
                        symbol=symbol,
                        anomaly_type=AnomalyType.VOLATILITY_ANOMALY,
                        severity=severity,
                        confidence=confidence,
                        description=f"Volatility regime change: {vol_changes.loc[idx]:.1%}",
                        features={
                            'volatility_change': vol_changes.loc[idx],
                            'old_volatility': data.loc[idx-1, 'volatility'] if idx > 0 else None,
                            'new_volatility': data.loc[idx, 'volatility']
                        }
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_microstructure_anomalies(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> List[Anomaly]:
        """Detect market microstructure anomalies"""
        anomalies = []
        
        # Calculate bid-ask spread proxy (if available)
        if 'bid' in data.columns and 'ask' in data.columns:
            data['spread'] = (data['ask'] - data['bid']) / data['close']
            spread_ma = data['spread'].rolling(window=self.config["lookback_period"]).mean()
            spread_spikes = data['spread'] > spread_ma * 2.0
            
            for idx in data[spread_spikes].index:
                spread_ratio = data.loc[idx, 'spread'] / spread_ma.loc[idx]
                severity = min((spread_ratio - 1) / 2.0, 1.0)
                confidence = 0.8
                
                if severity >= self.config["min_anomaly_severity"]:
                    anomaly = Anomaly(
                        timestamp=data.loc[idx, 'timestamp'],
                        symbol=symbol,
                        anomaly_type=AnomalyType.MICROSTRUCTURE_ANOMALY,
                        severity=severity,
                        confidence=confidence,
                        description=f"Spread spike: {spread_ratio:.2f}x average",
                        features={
                            'spread_ratio': spread_ratio,
                            'spread': data.loc[idx, 'spread'],
                            'volume': data.loc[idx, 'volume']
                        }
                    )
                    anomalies.append(anomaly)
        
        # Detect price gaps
        data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        gap_threshold = 0.02  # 2% gap
        
        large_gaps = data['gap'].abs() > gap_threshold
        for idx in data[large_gaps].index:
            gap_size = data.loc[idx, 'gap']
            severity = min(abs(gap_size) / 0.1, 1.0)
            confidence = 0.9
            
            if severity >= self.config["min_anomaly_severity"]:
                anomaly = Anomaly(
                    timestamp=data.loc[idx, 'timestamp'],
                    symbol=symbol,
                    anomaly_type=AnomalyType.MICROSTRUCTURE_ANOMALY,
                    severity=severity,
                    confidence=confidence,
                    description=f"Price gap: {gap_size:.2%}",
                    features={
                        'gap_size': gap_size,
                        'open': data.loc[idx, 'open'],
                        'prev_close': data.loc[idx-1, 'close'] if idx > 0 else None
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_momentum_anomalies(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> List[Anomaly]:
        """Detect momentum-based anomalies"""
        anomalies = []
        
        # Calculate momentum indicators
        for period in [5, 10, 20, 60]:
            data[f'momentum_{period}'] = data['close'].pct_change(period)
        
        # Detect momentum acceleration
        for period in [5, 10, 20]:
            momentum_change = data[f'momentum_{period}'].diff()
            acceleration_threshold = self.config["momentum_threshold"]
            
            acceleration_events = momentum_change.abs() > acceleration_threshold
            
            for idx in data[acceleration_events].index:
                if not pd.isna(momentum_change.loc[idx]):
                    severity = min(abs(momentum_change.loc[idx]) / 0.05, 1.0)
                    confidence = 0.7
                    
                    if severity >= self.config["min_anomaly_severity"]:
                        anomaly = Anomaly(
                            timestamp=data.loc[idx, 'timestamp'],
                            symbol=symbol,
                            anomaly_type=AnomalyType.MOMENTUM_ANOMALY,
                            severity=severity,
                            confidence=confidence,
                            description=f"Momentum acceleration ({period}d): {momentum_change.loc[idx]:.2%}",
                            features={
                                'momentum_change': momentum_change.loc[idx],
                                'period': period,
                                'current_momentum': data.loc[idx, f'momentum_{period}']
                            }
                        )
                        anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_correlation_anomalies(
        self,
        market_data: pd.DataFrame
    ) -> List[Anomaly]:
        """Detect cross-asset correlation anomalies"""
        anomalies = []
        
        # Calculate correlation matrix
        symbols = market_data['symbol'].unique()
        if len(symbols) < 2:
            return anomalies
        
        # Create returns matrix
        returns_data = {}
        for symbol in symbols:
            symbol_data = market_data[market_data['symbol'] == symbol].copy()
            symbol_data['returns'] = symbol_data['close'].pct_change()
            returns_data[symbol] = symbol_data.set_index('timestamp')['returns']
        
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if len(returns_df) < 60:  # Need sufficient data
            return anomalies
        
        # Calculate rolling correlations
        correlation_window = 60
        correlations = returns_df.rolling(correlation_window).corr()
        
        # Find correlation breakdowns
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                if symbol1 != symbol2:
                    pair_corr = correlations.loc[(slice(None), symbol1), symbol2]
                    
                    # Find significant correlation changes
                    corr_changes = pair_corr.diff().abs()
                    significant_changes = corr_changes > 0.3  # 30% correlation change
                    
                    for timestamp in pair_corr[significant_changes].index:
                        if isinstance(timestamp, tuple):
                            timestamp = timestamp[0]
                        
                        change_magnitude = corr_changes.loc[timestamp]
                        severity = min(change_magnitude / 0.5, 1.0)
                        confidence = 0.6
                        
                        if severity >= self.config["min_anomaly_severity"]:
                            anomaly = Anomaly(
                                timestamp=timestamp,
                                symbol=f"{symbol1}-{symbol2}",
                                anomaly_type=AnomalyType.CORRELATION_ANOMALY,
                                severity=severity,
                                confidence=confidence,
                                description=f"Correlation breakdown: {change_magnitude:.3f} change",
                                features={
                                    'correlation_change': change_magnitude,
                                    'symbol1': symbol1,
                                    'symbol2': symbol2,
                                    'current_correlation': pair_corr.loc[timestamp]
                                }
                            )
                            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_regime_changes(
        self,
        market_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Detect market regime changes"""
        regime_changes = []
        
        # Calculate market-wide metrics
        market_data['returns'] = market_data.groupby('symbol')['close'].pct_change()
        
        # Market volatility regime
        market_vol = market_data.groupby('timestamp')['returns'].std()
        vol_regime_changes = market_vol.pct_change().abs() > self.config["regime_change_threshold"]
        
        for timestamp in market_vol[vol_regime_changes].index:
            vol_change = market_vol.pct_change().loc[timestamp]
            regime_changes.append({
                'timestamp': timestamp,
                'type': 'volatility_regime',
                'magnitude': vol_change,
                'description': f"Volatility regime change: {vol_change:.1%}"
            })
        
        # Market correlation regime
        symbols = market_data['symbol'].unique()
        if len(symbols) > 1:
            returns_matrix = market_data.pivot(index='timestamp', columns='symbol', values='returns')
            correlation_matrix = returns_matrix.rolling(60).corr()
            
            # Calculate average correlation
            avg_corr = correlation_matrix.groupby(level=0).mean().mean(axis=1)
            corr_changes = avg_corr.diff().abs() > 0.1  # 10% correlation change
            
            for timestamp in avg_corr[corr_changes].index:
                corr_change = avg_corr.diff().loc[timestamp]
                regime_changes.append({
                    'timestamp': timestamp,
                    'type': 'correlation_regime',
                    'magnitude': corr_change,
                    'description': f"Correlation regime change: {corr_change:.3f}"
                })
        
        return regime_changes
    
    def _filter_anomalies(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """Filter anomalies based on severity and confidence thresholds"""
        filtered = []
        
        for anomaly in anomalies:
            if (anomaly.severity >= self.config["min_anomaly_severity"] and
                anomaly.confidence >= self.config["min_confidence"]):
                filtered.append(anomaly)
        
        # Sort by severity * confidence
        filtered.sort(key=lambda x: x.severity * x.confidence, reverse=True)
        
        return filtered
    
    def _analyze_persistence(self, anomalies: List[Anomaly]) -> Dict[str, float]:
        """Analyze persistence of different anomaly types"""
        persistence = {}
        
        for anomaly_type in AnomalyType:
            type_anomalies = [a for a in anomalies if a.anomaly_type == anomaly_type]
            
            if len(type_anomalies) > 0:
                # Calculate average severity and confidence
                avg_severity = np.mean([a.severity for a in type_anomalies])
                avg_confidence = np.mean([a.confidence for a in type_anomalies])
                
                # Calculate persistence score
                persistence_score = avg_severity * avg_confidence
                
                persistence[anomaly_type.value] = persistence_score
        
        return persistence
    
    def _calculate_anomaly_correlations(self, anomalies: List[Anomaly]) -> pd.DataFrame:
        """Calculate correlations between different anomaly types"""
        if len(anomalies) == 0:
            return pd.DataFrame()
        
        # Create anomaly matrix
        anomaly_data = []
        for anomaly in anomalies:
            anomaly_data.append({
                'timestamp': anomaly.timestamp,
                'type': anomaly.anomaly_type.value,
                'severity': anomaly.severity,
                'confidence': anomaly.confidence,
                'score': anomaly.severity * anomaly.confidence
            })
        
        anomaly_df = pd.DataFrame(anomaly_data)
        
        if len(anomaly_df) == 0:
            return pd.DataFrame()
        
        # Pivot to get anomaly types as columns
        pivot_df = anomaly_df.pivot_table(
            index='timestamp',
            columns='type',
            values='score',
            aggfunc='sum'
        ).fillna(0)
        
        # Calculate correlation matrix
        correlation_matrix = pivot_df.corr()
        
        return correlation_matrix
    
    def _calculate_summary_stats(self, anomalies: List[Anomaly]) -> Dict[str, float]:
        """Calculate summary statistics for anomalies"""
        if len(anomalies) == 0:
            return {}
        
        stats = {
            'total_anomalies': len(anomalies),
            'avg_severity': np.mean([a.severity for a in anomalies]),
            'avg_confidence': np.mean([a.confidence for a in anomalies]),
            'max_severity': max([a.severity for a in anomalies]),
            'min_severity': min([a.severity for a in anomalies])
        }
        
        # Count by type
        for anomaly_type in AnomalyType:
            type_count = len([a for a in anomalies if a.anomaly_type == anomaly_type])
            stats[f'{anomaly_type.value}_count'] = type_count
        
        return stats
    
    def _generate_recommendations(
        self,
        anomalies: List[Anomaly],
        persistence_analysis: Dict[str, float],
        correlation_matrix: pd.DataFrame
    ) -> List[str]:
        """Generate recommendations based on anomaly analysis"""
        recommendations = []
        
        if len(anomalies) == 0:
            recommendations.append("No significant anomalies detected in current data")
            return recommendations
        
        # Analyze most persistent anomalies
        if persistence_analysis:
            most_persistent = max(persistence_analysis.items(), key=lambda x: x[1])
            recommendations.append(f"Focus on {most_persistent[0]} anomalies (persistence: {most_persistent[1]:.3f})")
        
        # Analyze anomaly correlations
        if not correlation_matrix.empty:
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > self.config["correlation_threshold"]:
                        high_corr_pairs.append((
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            corr
                        ))
            
            if high_corr_pairs:
                recommendations.append("Consider combining highly correlated anomaly types")
                for pair in high_corr_pairs[:3]:  # Top 3 pairs
                    recommendations.append(f"  - {pair[0]} and {pair[1]} (correlation: {pair[2]:.3f})")
        
        # General recommendations
        recommendations.append("Focus on anomalies with high severity and confidence")
        recommendations.append("Monitor anomaly persistence over time")
        recommendations.append("Consider regime changes when interpreting anomalies")
        recommendations.append("Combine multiple anomaly types for robust signals")
        
        return recommendations


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = MarketAnomalyDetector()
    
    # Create sample market data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    market_data = []
    for symbol in symbols:
        # Generate realistic price data
        np.random.seed(42)
        price = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
        volume = np.random.randint(1000000, 10000000, len(dates))
        
        for i, date in enumerate(dates):
            market_data.append({
                'timestamp': date,
                'symbol': symbol,
                'open': price[i] * (1 + np.random.randn() * 0.01),
                'high': price[i] * (1 + abs(np.random.randn()) * 0.02),
                'low': price[i] * (1 - abs(np.random.randn()) * 0.02),
                'close': price[i],
                'volume': volume[i]
            })
    
    market_df = pd.DataFrame(market_data)
    
    # Find anomalies
    analysis = detector.find_anomalies(market_df)
    
    print("Anomaly Detection Results:")
    print(f"Total anomalies found: {len(analysis.anomalies)}")
    print(f"Summary statistics: {analysis.summary_stats}")
    
    print("\nTop 5 anomalies:")
    for i, anomaly in enumerate(analysis.anomalies[:5]):
        print(f"{i+1}. {anomaly.symbol} - {anomaly.anomaly_type.value}")
        print(f"   Severity: {anomaly.severity:.3f}, Confidence: {anomaly.confidence:.3f}")
        print(f"   Description: {anomaly.description}")
    
    print("\nPersistence analysis:")
    for anomaly_type, persistence in analysis.persistence_analysis.items():
        print(f"  {anomaly_type}: {persistence:.3f}")
    
    print("\nRecommendations:")
    for rec in analysis.recommendations:
        print(f"  • {rec}") 