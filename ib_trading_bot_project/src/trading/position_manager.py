"""
Position Manager

Tracks portfolio positions, calculates P&L, and manages position data.
"""

import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    from loguru import logger  # type: ignore
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: float
    last_price: float = 0.0

class PositionManager:
    """
    Manages portfolio positions and P&L calculations
    """
    
    def __init__(self, connector):
        """
        Initialize position manager
        
        Args:
            connector: IB connector instance
        """
        self.connector = connector
        self.positions: Dict[str, Position] = {}
        self._lock = threading.Lock()
        self._last_update = 0.0
        
        # Set up position callback
        if hasattr(connector, 'position_callback'):
            connector.position_callback = self._on_position_update
    
    def update_positions(self):
        """Update all position data"""
        try:
            # Get current positions from connector
            ib_positions = self.connector.get_positions()
            
            with self._lock:
                # Update positions
                for symbol, pos_data in ib_positions.items():
                    self._update_position(symbol, pos_data)
                
                self._last_update = time.time()
            
            logger.debug(f"Updated {len(ib_positions)} positions")
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _update_position(self, symbol: str, pos_data: Dict[str, Any]):
        """Update a single position"""
        try:
            quantity = pos_data.get("position", 0)
            avg_cost = pos_data.get("avg_cost", 0.0)
            
            if quantity == 0:
                # Remove zero positions
                if symbol in self.positions:
                    del self.positions[symbol]
                return
            
            # Get current market price
            market_data = self.connector.get_market_data(symbol)
            last_price = market_data.get("last_price", 0.0) if market_data else 0.0
            
            # Calculate market value and P&L
            market_value = quantity * last_price
            unrealized_pnl = market_value - (quantity * avg_cost)
            
            # Create or update position
            if symbol in self.positions:
                position = self.positions[symbol]
                position.quantity = quantity
                position.avg_cost = avg_cost
                position.market_value = market_value
                position.unrealized_pnl = unrealized_pnl
                position.last_price = last_price
                position.timestamp = time.time()
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost=avg_cost,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=0.0,
                    timestamp=time.time(),
                    last_price=last_price
                )
                
        except Exception as e:
            logger.error(f"Error updating position for {symbol}: {e}")
    
    def _on_position_update(self, symbol: str, quantity: int, avg_cost: float):
        """Handle position updates from connector"""
        try:
            pos_data = {
                "position": quantity,
                "avg_cost": avg_cost
            }
            self._update_position(symbol, pos_data)
            
        except Exception as e:
            logger.error(f"Error processing position update: {e}")
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position object or None if not found
        """
        with self._lock:
            return self.positions.get(symbol)
    
    def get_positions(self) -> List[Position]:
        """Get all positions"""
        with self._lock:
            return list(self.positions.values())
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions"""
        with self._lock:
            total_positions = len(self.positions)
            total_quantity = 0
            total_market_value = 0.0
            total_unrealized_pnl = 0.0
            total_realized_pnl = 0.0
            
            for position in self.positions.values():
                total_quantity += abs(position.quantity)
                total_market_value += position.market_value
                total_unrealized_pnl += position.unrealized_pnl
                total_realized_pnl += position.realized_pnl
            
            return {
                "total_positions": total_positions,
                "total_quantity": total_quantity,
                "total_market_value": total_market_value,
                "total_unrealized_pnl": total_unrealized_pnl,
                "total_realized_pnl": total_realized_pnl,
                "total_pnl": total_unrealized_pnl + total_realized_pnl,
                "last_update": self._last_update
            }
    
    def get_largest_positions(self, count: int = 5) -> List[Position]:
        """
        Get largest positions by market value
        
        Args:
            count: Number of positions to return
            
        Returns:
            List of largest positions
        """
        with self._lock:
            sorted_positions = sorted(
                self.positions.values(),
                key=lambda x: abs(x.market_value),
                reverse=True
            )
            return sorted_positions[:count]
    
    def get_profitable_positions(self) -> List[Position]:
        """Get all profitable positions"""
        with self._lock:
            return [pos for pos in self.positions.values() if pos.unrealized_pnl > 0]
    
    def get_losing_positions(self) -> List[Position]:
        """Get all losing positions"""
        with self._lock:
            return [pos for pos in self.positions.values() if pos.unrealized_pnl < 0]
    
    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """
        Get position by symbol (case-insensitive)
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position object or None if not found
        """
        symbol_upper = symbol.upper()
        with self._lock:
            return self.positions.get(symbol_upper)
    
    def has_position(self, symbol: str) -> bool:
        """
        Check if we have a position in a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if position exists
        """
        return self.get_position(symbol) is not None
    
    def get_position_size(self, symbol: str) -> int:
        """
        Get position size for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position size (positive for long, negative for short)
        """
        position = self.get_position(symbol)
        return position.quantity if position else 0
    
    def get_position_value(self, symbol: str) -> float:
        """
        Get market value of position for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Market value
        """
        position = self.get_position(symbol)
        return position.market_value if position else 0.0
    
    def get_position_pnl(self, symbol: str) -> float:
        """
        Get unrealized P&L for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Unrealized P&L
        """
        position = self.get_position(symbol)
        return position.unrealized_pnl if position else 0.0
    
    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio performance metrics"""
        summary = self.get_position_summary()
        
        if summary["total_market_value"] > 0:
            # Calculate percentage returns
            total_pnl = summary["total_pnl"]
            pnl_percentage = (total_pnl / summary["total_market_value"]) * 100
            
            # Calculate position concentration
            positions = self.get_positions()
            if positions:
                largest_position = max(positions, key=lambda x: abs(x.market_value))
                concentration = (abs(largest_position.market_value) / summary["total_market_value"]) * 100
            else:
                concentration = 0.0
            
            metrics = {
                "total_pnl": total_pnl,
                "pnl_percentage": pnl_percentage,
                "position_concentration": concentration,
                "profitable_positions": len(self.get_profitable_positions()),
                "losing_positions": len(self.get_losing_positions()),
                "total_positions": summary["total_positions"]
            }
        else:
            metrics = {
                "total_pnl": 0.0,
                "pnl_percentage": 0.0,
                "position_concentration": 0.0,
                "profitable_positions": 0,
                "losing_positions": 0,
                "total_positions": 0
            }
        
        return metrics
    
    def export_positions(self, filename: str) -> bool:
        """
        Export positions to CSV file
        
        Args:
            filename: Output filename
            
        Returns:
            True if successful
        """
        try:
            import csv
            
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['symbol', 'quantity', 'avg_cost', 'market_value', 
                             'unrealized_pnl', 'realized_pnl', 'last_price', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for position in self.get_positions():
                    writer.writerow({
                        'symbol': position.symbol,
                        'quantity': position.quantity,
                        'avg_cost': position.avg_cost,
                        'market_value': position.market_value,
                        'unrealized_pnl': position.unrealized_pnl,
                        'realized_pnl': position.realized_pnl,
                        'last_price': position.last_price,
                        'timestamp': position.timestamp
                    })
            
            logger.info(f"Positions exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting positions: {e}")
            return False 