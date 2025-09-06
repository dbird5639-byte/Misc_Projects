"""
Helper functions for trading operations

Common utilities used throughout the trading bot.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

def validate_symbol(symbol: str) -> bool:
    """
    Validate stock symbol format
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Remove whitespace and convert to uppercase
    symbol = symbol.strip().upper()
    
    # Check length (1-5 characters for most stocks)
    if len(symbol) < 1 or len(symbol) > 5:
        return False
    
    # Check for valid characters (letters only)
    if not re.match(r'^[A-Z]+$', symbol):
        return False
    
    return True

def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format currency amount
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def calculate_returns(initial_value: float, final_value: float) -> Dict[str, float]:
    """
    Calculate return metrics
    
    Args:
        initial_value: Initial value
        final_value: Final value
        
    Returns:
        Dictionary of return metrics
    """
    if initial_value <= 0:
        return {
            "absolute_return": 0.0,
            "percentage_return": 0.0,
            "log_return": 0.0
        }
    
    absolute_return = final_value - initial_value
    percentage_return = (absolute_return / initial_value) * 100
    
    # Log return for better statistical properties
    log_return = 0.0
    if final_value > 0:
        log_return = (final_value / initial_value) ** 0.5 - 1
    
    return {
        "absolute_return": absolute_return,
        "percentage_return": percentage_return,
        "log_return": log_return
    }

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    # Calculate average return
    avg_return = sum(returns) / len(returns)
    
    # Calculate standard deviation
    variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
    std_dev = variance ** 0.5
    
    if std_dev == 0:
        return 0.0
    
    # Annualize (assuming daily returns)
    annualized_return = avg_return * 252
    annualized_std = std_dev * (252 ** 0.5)
    
    sharpe = (annualized_return - risk_free_rate) / annualized_std
    
    return sharpe

def calculate_max_drawdown(values: List[float]) -> Dict[str, float]:
    """
    Calculate maximum drawdown
    
    Args:
        values: List of portfolio values
        
    Returns:
        Dictionary with max drawdown metrics
    """
    if not values or len(values) < 2:
        return {
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "drawdown_duration": 0
        }
    
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    peak = values[0]
    peak_index = 0
    drawdown_start = 0
    drawdown_end = 0
    
    for i, value in enumerate(values):
        if value > peak:
            peak = value
            peak_index = i
        else:
            drawdown = peak - value
            drawdown_pct = (drawdown / peak) * 100
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
                drawdown_start = peak_index
                drawdown_end = i
    
    drawdown_duration = drawdown_end - drawdown_start
    
    return {
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "drawdown_duration": drawdown_duration
    }

def calculate_position_size(portfolio_value: float, price: float, 
                          risk_per_trade: float = 0.02) -> int:
    """
    Calculate position size based on risk
    
    Args:
        portfolio_value: Total portfolio value
        price: Stock price
        risk_per_trade: Risk per trade as percentage
        
    Returns:
        Number of shares
    """
    if price <= 0:
        return 0
    
    risk_amount = portfolio_value * risk_per_trade
    shares = int(risk_amount / price)
    
    return max(0, shares)

def format_timestamp(timestamp: float) -> str:
    """
    Format timestamp to readable string
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        Formatted timestamp string
    """
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

def parse_timestamp(timestamp_str: str) -> float:
    """
    Parse timestamp string to Unix timestamp
    
    Args:
        timestamp_str: Timestamp string
        
    Returns:
        Unix timestamp
    """
    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        return dt.timestamp()
    except ValueError:
        return 0.0

def calculate_correlation(series1: List[float], series2: List[float]) -> float:
    """
    Calculate correlation between two series
    
    Args:
        series1: First series
        series2: Second series
        
    Returns:
        Correlation coefficient
    """
    if len(series1) != len(series2) or len(series1) < 2:
        return 0.0
    
    # Calculate means
    mean1 = sum(series1) / len(series1)
    mean2 = sum(series2) / len(series2)
    
    # Calculate covariance and variances
    covariance = sum((x - mean1) * (y - mean2) for x, y in zip(series1, series2))
    variance1 = sum((x - mean1) ** 2 for x in series1)
    variance2 = sum((y - mean2) ** 2 for y in series2)
    
    # Calculate correlation
    if variance1 == 0 or variance2 == 0:
        return 0.0
    
    correlation = covariance / (variance1 ** 0.5 * variance2 ** 0.5)
    
    return max(-1.0, min(1.0, correlation))

def calculate_volatility(returns: List[float], annualize: bool = True) -> float:
    """
    Calculate volatility
    
    Args:
        returns: List of returns
        annualize: Whether to annualize the volatility
        
    Returns:
        Volatility
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    # Calculate mean return
    mean_return = sum(returns) / len(returns)
    
    # Calculate variance
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    volatility = variance ** 0.5
    
    # Annualize if requested
    if annualize:
        volatility *= (252 ** 0.5)  # Assuming daily returns
    
    return volatility

def calculate_beta(asset_returns: List[float], market_returns: List[float]) -> float:
    """
    Calculate beta relative to market
    
    Args:
        asset_returns: Asset returns
        market_returns: Market returns
        
    Returns:
        Beta value
    """
    if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
        return 1.0
    
    # Calculate correlation
    correlation = calculate_correlation(asset_returns, market_returns)
    
    # Calculate volatilities
    asset_vol = calculate_volatility(asset_returns, annualize=False)
    market_vol = calculate_volatility(market_returns, annualize=False)
    
    if market_vol == 0:
        return 1.0
    
    beta = correlation * (asset_vol / market_vol)
    
    return beta

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage
    
    Args:
        value: Value to format
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimals}f}%"

def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate Kelly Criterion for position sizing
    
    Args:
        win_rate: Win rate (0-1)
        avg_win: Average win size
        avg_loss: Average loss size
        
    Returns:
        Kelly fraction
    """
    if avg_loss == 0:
        return 0.0
    
    kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    
    # Cap at reasonable levels
    return max(0.0, min(0.25, kelly))

def calculate_risk_adjusted_return(returns: List[float], risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate various risk-adjusted return metrics
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary of risk-adjusted return metrics
    """
    if not returns:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "information_ratio": 0.0
        }
    
    # Calculate basic metrics
    avg_return = sum(returns) / len(returns)
    volatility = calculate_volatility(returns, annualize=False)
    
    # Sharpe ratio
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    
    # Sortino ratio (downside deviation)
    downside_returns = [r for r in returns if r < avg_return]
    if downside_returns:
        downside_deviation = calculate_volatility(downside_returns, annualize=False)
        sortino = (avg_return - risk_free_rate/252) / downside_deviation if downside_deviation > 0 else 0.0
    else:
        sortino = 0.0
    
    # Calmar ratio (return / max drawdown)
    values = [1.0]
    for r in returns:
        values.append(values[-1] * (1 + r))
    
    drawdown_info = calculate_max_drawdown(values)
    max_dd_pct = drawdown_info["max_drawdown_pct"] / 100
    
    calmar = avg_return / max_dd_pct if max_dd_pct > 0 else 0.0
    
    # Information ratio (excess return / tracking error)
    # Simplified - assumes benchmark return of 0
    information_ratio = avg_return / volatility if volatility > 0 else 0.0
    
    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "information_ratio": information_ratio
    }

def validate_price(price: float) -> bool:
    """
    Validate price value
    
    Args:
        price: Price to validate
        
    Returns:
        True if valid, False otherwise
    """
    return isinstance(price, (int, float)) and price > 0

def validate_quantity(quantity: int) -> bool:
    """
    Validate quantity value
    
    Args:
        quantity: Quantity to validate
        
    Returns:
        True if valid, False otherwise
    """
    return isinstance(quantity, int) and quantity > 0

def round_to_tick_size(price: float, tick_size: float = 0.01) -> float:
    """
    Round price to tick size
    
    Args:
        price: Price to round
        tick_size: Tick size
        
    Returns:
        Rounded price
    """
    if tick_size <= 0:
        return price
    
    return round(price / tick_size) * tick_size

def calculate_commission(quantity: int, price: float, commission_rate: float = 0.005) -> float:
    """
    Calculate commission for a trade
    
    Args:
        quantity: Number of shares
        price: Price per share
        commission_rate: Commission rate per share
        
    Returns:
        Commission amount
    """
    return quantity * price * commission_rate

def calculate_slippage(quantity: int, price: float, slippage_rate: float = 0.001) -> float:
    """
    Calculate estimated slippage
    
    Args:
        quantity: Number of shares
        price: Price per share
        slippage_rate: Slippage rate
        
    Returns:
        Slippage amount
    """
    return quantity * price * slippage_rate 