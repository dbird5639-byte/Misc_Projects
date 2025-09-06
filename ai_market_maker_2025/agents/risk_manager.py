"""
Risk Manager - AI agent for managing portfolio risk and position sizing
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import plotly.express as px

from ..config.settings import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskMetrics:
    """Represents portfolio risk metrics"""
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    volatility: float
    beta: float
    correlation: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class PositionRisk:
    """Represents risk assessment for a position"""
    symbol: str
    position_size: float
    current_value: float
    unrealized_pnl: float
    risk_score: float  # 0-1 scale
    var_contribution: float
    correlation_risk: float
    liquidity_risk: float
    volatility_risk: float
    recommended_action: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class RiskAlert:
    """Represents a risk alert"""
    alert_type: str  # 'high_risk', 'concentration', 'correlation', 'liquidity'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    affected_positions: List[str]
    recommended_actions: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class RiskManager:
    """
    AI agent for managing portfolio risk and position sizing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = True
        self.is_running = False
        
        # Settings
        self.max_portfolio_risk = config.get("max_portfolio_risk", 0.02)  # 2%
        self.max_position_risk = config.get("max_position_risk", 0.005)  # 0.5%
        self.max_correlation = config.get("max_correlation", 0.8)
        self.min_liquidity = config.get("min_liquidity", 1_000_000)
        self.risk_update_interval = config.get("risk_update_interval", 60)  # seconds
        self.alert_threshold = config.get("alert_threshold", 0.8)
        
        # AI Models
        self.anomaly_detector: Optional[IsolationForest] = None
        self.var_model = None
        self.correlation_model = None
        
        # Data storage
        self.portfolio_positions: Dict[str, Dict[str, Any]] = {}
        self.risk_metrics: List[RiskMetrics] = []
        self.position_risks: List[PositionRisk] = []
        self.risk_alerts: List[RiskAlert] = []
        self.historical_returns: List[float] = []
        
        # Performance metrics
        self.metrics = {
            "total_risk_assessments": 0,
            "alerts_generated": 0,
            "risk_mitigations": 0,
            "accuracy": 0.0,
            "processing_time": 0.0,
            "last_update": None
        }
        
        # Task management
        self.risk_task: Optional[asyncio.Task] = None
        self.alert_task: Optional[asyncio.Task] = None
        self.mitigation_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the risk manager"""
        try:
            logger.info("Initializing Risk Manager...")
            
            # Initialize AI models
            await self._initialize_models()
            
            # Load historical data
            await self._load_historical_data()
            
            # Start risk management tasks
            self.risk_task = asyncio.create_task(self._risk_assessment_loop())
            self.alert_task = asyncio.create_task(self._alert_monitoring_loop())
            self.mitigation_task = asyncio.create_task(self._risk_mitigation_loop())
            
            self.is_running = True
            logger.info("Risk Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Risk Manager: {e}")
            return False
    
    async def _initialize_models(self):
        """Initialize AI models"""
        try:
            # Initialize anomaly detector
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Initialize VaR model
            self.var_model = self._create_var_model()
            
            # Initialize correlation model
            self.correlation_model = self._create_correlation_model()
            
            logger.info("AI models initialized for risk management")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _create_var_model(self):
        """Create Value at Risk model"""
        # Simple historical VaR model
        return {
            "method": "historical",
            "confidence_levels": [0.95, 0.99],
            "window_size": 252  # 1 year of trading days
        }
    
    def _create_correlation_model(self):
        """Create correlation analysis model"""
        return {
            "method": "rolling_correlation",
            "window_size": 30,  # 30 days
            "min_correlation": 0.3
        }
    
    async def _risk_assessment_loop(self):
        """Main risk assessment loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Update portfolio positions
                await self._update_portfolio_positions()
                
                # Calculate risk metrics
                risk_metrics = await self._calculate_risk_metrics()
                self.risk_metrics.append(risk_metrics)
                
                # Assess position risks
                position_risks = await self._assess_position_risks()
                self.position_risks.extend(position_risks)
                
                # Detect anomalies
                await self._detect_anomalies()
                
                # Update metrics
                self.metrics["total_risk_assessments"] += 1
                self.metrics["processing_time"] = time.time() - start_time
                self.metrics["last_update"] = datetime.now()
                
                # Wait for next risk assessment cycle
                await asyncio.sleep(self.risk_update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in risk assessment loop: {e}")
                await asyncio.sleep(30)
    
    async def _alert_monitoring_loop(self):
        """Risk alert monitoring loop"""
        while self.is_running:
            try:
                # Generate risk alerts
                alerts = await self._generate_risk_alerts()
                self.risk_alerts.extend(alerts)
                
                # Send alerts if needed
                await self._send_risk_alerts(alerts)
                
                # Wait for next alert cycle (every 5 minutes)
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _risk_mitigation_loop(self):
        """Risk mitigation loop"""
        while self.is_running:
            try:
                # Check for high-risk situations
                high_risk_alerts = [a for a in self.risk_alerts 
                                  if a.severity in ['high', 'critical'] and
                                  a.timestamp > datetime.now() - timedelta(minutes=5)]
                
                # Execute risk mitigations
                for alert in high_risk_alerts:
                    await self._execute_risk_mitigation(alert)
                
                # Wait for next mitigation cycle (every 2 minutes)
                await asyncio.sleep(120)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in risk mitigation loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_portfolio_positions(self):
        """Update portfolio positions"""
        try:
            # This would integrate with position manager
            # For now, use mock data
            self.portfolio_positions = {
                "BTC": {
                    "size": 1.0,
                    "entry_price": 45000,
                    "current_price": 46000,
                    "unrealized_pnl": 1000,
                    "leverage": 2.0
                },
                "ETH": {
                    "size": 10.0,
                    "entry_price": 3000,
                    "current_price": 3100,
                    "unrealized_pnl": 1000,
                    "leverage": 1.5
                }
            }
            
        except Exception as e:
            logger.error(f"Error updating portfolio positions: {e}")
    
    async def _calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate portfolio risk metrics"""
        try:
            # Calculate returns
            returns = self._calculate_returns()
            
            if len(returns) < 2:
                return self._create_default_risk_metrics()
            
            # Calculate VaR
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Calculate volatility
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Calculate Sharpe ratio
            risk_free_rate = 0.02  # 2% annual
            excess_returns = np.array(returns) - risk_free_rate / 252
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
            # Calculate Sortino ratio
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_deviation = np.std(downside_returns)
                sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(252)
            else:
                sortino_ratio = float('inf')
            
            # Calculate maximum drawdown
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # Calculate Calmar ratio
            if max_drawdown != 0:
                calmar_ratio = np.mean(returns) * 252 / abs(max_drawdown)
            else:
                calmar_ratio = float('inf')
            
            # Calculate beta and correlation
            beta, correlation = self._calculate_beta_correlation(returns)
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                volatility=volatility,
                beta=beta,
                correlation=correlation,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return self._create_default_risk_metrics()
    
    def _create_default_risk_metrics(self) -> RiskMetrics:
        """Create default risk metrics"""
        return RiskMetrics(
            var_95=0.0,
            var_99=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            volatility=0.0,
            beta=1.0,
            correlation=0.0,
            timestamp=datetime.now()
        )
    
    def _calculate_returns(self) -> List[float]:
        """Calculate portfolio returns"""
        try:
            # This would calculate actual returns from position changes
            # For now, use mock data
            if len(self.historical_returns) > 0:
                return self.historical_returns[-252:]  # Last year
            else:
                # Generate mock returns
                returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
                self.historical_returns.extend(returns)
                return returns.tolist()
                
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return []
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return float(np.min(drawdown))
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_beta_correlation(self, returns: List[float]) -> Tuple[float, float]:
        """Calculate beta and correlation with market"""
        try:
            # This would use actual market data
            # For now, use mock market returns
            market_returns = np.random.normal(0.0005, 0.015, len(returns))
            
            # Calculate beta
            covariance = np.cov(returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance > 0 else 1.0
            
            # Calculate correlation
            correlation = np.corrcoef(returns, market_returns)[0, 1]
            
            return float(beta), float(correlation)
            
        except Exception as e:
            logger.error(f"Error calculating beta/correlation: {e}")
            return 1.0, 0.0
    
    async def _assess_position_risks(self) -> List[PositionRisk]:
        """Assess risks for individual positions"""
        position_risks = []
        
        try:
            for symbol, position in self.portfolio_positions.items():
                # Calculate various risk metrics
                risk_score = self._calculate_position_risk_score(position)
                var_contribution = self._calculate_var_contribution(symbol, position)
                correlation_risk = self._calculate_correlation_risk(symbol)
                liquidity_risk = self._calculate_liquidity_risk(symbol)
                volatility_risk = self._calculate_volatility_risk(symbol)
                
                # Determine recommended action
                recommended_action = self._determine_position_action(
                    risk_score, var_contribution, correlation_risk, liquidity_risk, volatility_risk
                )
                
                position_risk = PositionRisk(
                    symbol=symbol,
                    position_size=position.get("size", 0),
                    current_value=position.get("current_price", 0) * position.get("size", 0),
                    unrealized_pnl=position.get("unrealized_pnl", 0),
                    risk_score=risk_score,
                    var_contribution=var_contribution,
                    correlation_risk=correlation_risk,
                    liquidity_risk=liquidity_risk,
                    volatility_risk=volatility_risk,
                    recommended_action=recommended_action,
                    timestamp=datetime.now()
                )
                
                position_risks.append(position_risk)
            
        except Exception as e:
            logger.error(f"Error assessing position risks: {e}")
        
        return position_risks
    
    def _calculate_position_risk_score(self, position: Dict[str, Any]) -> float:
        """Calculate overall risk score for a position"""
        try:
            size = position.get("size", 0)
            leverage = position.get("leverage", 1.0)
            pnl = position.get("unrealized_pnl", 0)
            
            # Size risk (0-1)
            size_risk = min(size / 1000000, 1.0)  # Normalize to $1M
            
            # Leverage risk (0-1)
            leverage_risk = min(leverage / 10, 1.0)  # Normalize to 10x leverage
            
            # PnL risk (0-1)
            pnl_risk = min(abs(pnl) / 10000, 1.0)  # Normalize to $10K
            
            # Weighted average
            risk_score = (size_risk * 0.4 + leverage_risk * 0.4 + pnl_risk * 0.2)
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating position risk score: {e}")
            return 0.5
    
    def _calculate_var_contribution(self, symbol: str, position: Dict[str, Any]) -> float:
        """Calculate VaR contribution of a position"""
        try:
            # This would use actual VaR calculations
            # For now, use simplified approach
            position_value = position.get("size", 0) * position.get("current_price", 0)
            portfolio_value = sum(p.get("size", 0) * p.get("current_price", 0) 
                                for p in self.portfolio_positions.values())
            
            if portfolio_value > 0:
                return position_value / portfolio_value
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating VaR contribution: {e}")
            return 0.0
    
    def _calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation risk for a position"""
        try:
            # This would use actual correlation data
            # For now, return mock value
            return 0.3
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {e}")
            return 0.5
    
    def _calculate_liquidity_risk(self, symbol: str) -> float:
        """Calculate liquidity risk for a position"""
        try:
            # This would use actual liquidity data
            # For now, return mock value
            return 0.2
            
        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {e}")
            return 0.5
    
    def _calculate_volatility_risk(self, symbol: str) -> float:
        """Calculate volatility risk for a position"""
        try:
            # This would use actual volatility data
            # For now, return mock value
            return 0.4
            
        except Exception as e:
            logger.error(f"Error calculating volatility risk: {e}")
            return 0.5
    
    def _determine_position_action(self, risk_score: float, var_contribution: float, 
                                 correlation_risk: float, liquidity_risk: float, 
                                 volatility_risk: float) -> str:
        """Determine recommended action for a position"""
        try:
            # High risk scenarios
            if risk_score > 0.8 or var_contribution > 0.3:
                return "reduce_position"
            elif risk_score > 0.6:
                return "add_hedge"
            elif correlation_risk > 0.7:
                return "diversify"
            elif liquidity_risk > 0.7:
                return "reduce_size"
            elif volatility_risk > 0.7:
                return "add_stop_loss"
            else:
                return "hold"
                
        except Exception as e:
            logger.error(f"Error determining position action: {e}")
            return "hold"
    
    async def _detect_anomalies(self):
        """Detect anomalous risk patterns"""
        try:
            if len(self.position_risks) < 10:
                return
            
            # Extract features for anomaly detection
            features = []
            for risk in self.position_risks[-100:]:  # Last 100 assessments
                features.append([
                    risk.risk_score,
                    risk.var_contribution,
                    risk.correlation_risk,
                    risk.liquidity_risk,
                    risk.volatility_risk
                ])
            
            if len(features) > 0:
                # Fit anomaly detector
                self.anomaly_detector.fit(features)
                
                # Detect anomalies
                anomalies = self.anomaly_detector.predict(features)
                anomaly_indices = [i for i, a in enumerate(anomalies) if a == -1]
                
                if anomaly_indices:
                    logger.warning(f"Detected {len(anomaly_indices)} anomalous risk patterns")
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
    
    async def _generate_risk_alerts(self) -> List[RiskAlert]:
        """Generate risk alerts"""
        alerts = []
        
        try:
            # Check portfolio risk
            if self.risk_metrics:
                latest_metrics = self.risk_metrics[-1]
                
                # High VaR alert
                if latest_metrics.var_95 < -self.max_portfolio_risk:
                    alerts.append(RiskAlert(
                        alert_type="high_risk",
                        severity="critical",
                        message=f"Portfolio VaR exceeded limit: {latest_metrics.var_95:.2%}",
                        affected_positions=list(self.portfolio_positions.keys()),
                        recommended_actions=["reduce_positions", "add_hedges"],
                        timestamp=datetime.now()
                    ))
                
                # High correlation alert
                if latest_metrics.correlation > self.max_correlation:
                    alerts.append(RiskAlert(
                        alert_type="correlation",
                        severity="high",
                        message=f"High portfolio correlation: {latest_metrics.correlation:.2f}",
                        affected_positions=list(self.portfolio_positions.keys()),
                        recommended_actions=["diversify_portfolio"],
                        timestamp=datetime.now()
                    ))
            
            # Check position risks
            for risk in self.position_risks[-10:]:  # Last 10 assessments
                if risk.risk_score > self.alert_threshold:
                    alerts.append(RiskAlert(
                        alert_type="high_risk",
                        severity="high",
                        message=f"High risk position: {risk.symbol} (score: {risk.risk_score:.2f})",
                        affected_positions=[risk.symbol],
                        recommended_actions=[risk.recommended_action],
                        timestamp=datetime.now()
                    ))
            
            self.metrics["alerts_generated"] += len(alerts)
            
        except Exception as e:
            logger.error(f"Error generating risk alerts: {e}")
        
        return alerts
    
    async def _send_risk_alerts(self, alerts: List[RiskAlert]):
        """Send risk alerts"""
        try:
            for alert in alerts:
                if alert.severity in ['high', 'critical']:
                    logger.warning(f"Risk Alert: {alert.message}")
                    # This would send to notification system
                    
        except Exception as e:
            logger.error(f"Error sending risk alerts: {e}")
    
    async def _execute_risk_mitigation(self, alert: RiskAlert):
        """Execute risk mitigation actions"""
        try:
            logger.info(f"Executing risk mitigation for: {alert.message}")
            
            # This would execute actual trading actions
            # For now, just log the action
            
            self.metrics["risk_mitigations"] += 1
            
        except Exception as e:
            logger.error(f"Error executing risk mitigation: {e}")
    
    async def _load_historical_data(self):
        """Load historical risk data"""
        try:
            # Load from database or file
            logger.info("Loading historical risk data...")
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def get_current_risk_metrics(self) -> Optional[RiskMetrics]:
        """Get current risk metrics"""
        return self.risk_metrics[-1] if self.risk_metrics else None
    
    def get_position_risks(self) -> List[PositionRisk]:
        """Get current position risks"""
        return self.position_risks[-len(self.portfolio_positions):] if self.position_risks else []
    
    def get_active_alerts(self, hours: int = 24) -> List[RiskAlert]:
        """Get active risk alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [a for a in self.risk_alerts if a.timestamp > cutoff_time]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics
    
    async def stop(self):
        """Stop the risk manager"""
        self.is_running = False
        
        if self.risk_task:
            self.risk_task.cancel()
        
        if self.alert_task:
            self.alert_task.cancel()
        
        if self.mitigation_task:
            self.mitigation_task.cancel()
        
        logger.info("Risk Manager stopped")


def create_risk_manager(config: Dict[str, Any]) -> RiskManager:
    """Create a new risk manager instance"""
    return RiskManager(config) 