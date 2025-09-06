"""
Order Manager

Handles order placement, management, and execution tracking.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

try:
    from loguru import logger  # type: ignore
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

@dataclass
class Order:
    """Order data structure"""
    order_id: int
    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: str  # 'MKT', 'LMT', 'STP', etc.
    price: float
    status: str  # 'SUBMITTED', 'FILLED', 'CANCELLED', 'REJECTED'
    timestamp: float
    fill_price: Optional[float] = None
    filled_quantity: int = 0
    remaining_quantity: int = 0

class OrderManager:
    """
    Manages order placement, tracking, and execution
    """
    
    def __init__(self, connector):
        """
        Initialize order manager
        
        Args:
            connector: IB connector instance
        """
        self.connector = connector
        self.orders: Dict[int, Order] = {}
        self.next_order_id = 1
        self._lock = threading.Lock()
        self._callbacks: List[Callable] = []
        
        # Set up order status callback
        if hasattr(connector, 'order_callback'):
            connector.order_callback = self._on_order_update
    
    def place_market_order(self, symbol: str, action: str, 
                          quantity: int) -> int:
        """
        Place a market order
        
        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares
            
        Returns:
            Order ID
        """
        return self.place_order(symbol, action, quantity, "MKT")
    
    def place_limit_order(self, symbol: str, action: str, 
                         quantity: int, price: float) -> int:
        """
        Place a limit order
        
        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares
            price: Limit price
            
        Returns:
            Order ID
        """
        return self.place_order(symbol, action, quantity, "LMT", price)
    
    def place_stop_order(self, symbol: str, action: str, 
                        quantity: int, stop_price: float) -> int:
        """
        Place a stop order
        
        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares
            stop_price: Stop price
            
        Returns:
            Order ID
        """
        return self.place_order(symbol, action, quantity, "STP", stop_price)
    
    def place_order(self, symbol: str, action: str, quantity: int,
                   order_type: str, price: float = 0.0) -> int:
        """
        Place an order
        
        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares
            order_type: Order type
            price: Price (for limit/stop orders)
            
        Returns:
            Order ID
        """
        try:
            logger.info(f"Placing {order_type} {action} order for {quantity} {symbol}")
            
            # Validate order parameters
            if not self._validate_order(symbol, action, quantity, order_type, price):
                return -1
            
            # Place order through connector
            order_id = self.connector.place_order(symbol, action, quantity, order_type, price)
            
            if order_id > 0:
                # Create order record
                order = Order(
                    order_id=order_id,
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    order_type=order_type,
                    price=price,
                    status="SUBMITTED",
                    timestamp=time.time(),
                    remaining_quantity=quantity
                )
                
                with self._lock:
                    self.orders[order_id] = order
                
                logger.info(f"Order {order_id} placed successfully")
                return order_id
            else:
                logger.error("Failed to place order")
                return -1
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return -1
    
    def _validate_order(self, symbol: str, action: str, quantity: int,
                       order_type: str, price: float) -> bool:
        """Validate order parameters"""
        if not symbol or not symbol.strip():
            logger.error("Invalid symbol")
            return False
        
        if action not in ["BUY", "SELL"]:
            logger.error("Invalid action - must be BUY or SELL")
            return False
        
        if quantity <= 0:
            logger.error("Quantity must be positive")
            return False
        
        if order_type not in ["MKT", "LMT", "STP", "STP LMT"]:
            logger.error("Invalid order type")
            return False
        
        if order_type in ["LMT", "STP"] and price <= 0:
            logger.error("Price must be positive for limit/stop orders")
            return False
        
        return True
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful
        """
        try:
            logger.info(f"Cancelling order {order_id}")
            
            with self._lock:
                if order_id not in self.orders:
                    logger.error(f"Order {order_id} not found")
                    return False
                
                order = self.orders[order_id]
                if order.status in ["FILLED", "CANCELLED", "REJECTED"]:
                    logger.warning(f"Order {order_id} cannot be cancelled (status: {order.status})")
                    return False
            
            # Cancel through connector
            success = self.connector.cancel_order(order_id)
            
            if success:
                with self._lock:
                    self.orders[order_id].status = "CANCELLED"
                
                logger.info(f"Order {order_id} cancelled successfully")
                return True
            else:
                logger.error(f"Failed to cancel order {order_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_order(self, order_id: int) -> Optional[Order]:
        """
        Get order by ID
        
        Args:
            order_id: Order ID
            
        Returns:
            Order object or None if not found
        """
        with self._lock:
            return self.orders.get(order_id)
    
    def get_orders(self, status: Optional[str] = None) -> List[Order]:
        """
        Get orders with optional status filter
        
        Args:
            status: Filter by status (optional)
            
        Returns:
            List of orders
        """
        with self._lock:
            if status:
                return [order for order in self.orders.values() if order.status == status]
            else:
                return list(self.orders.values())
    
    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders"""
        return self.get_orders("SUBMITTED")
    
    def get_filled_orders(self) -> List[Order]:
        """Get all filled orders"""
        return self.get_orders("FILLED")
    
    def get_order_summary(self) -> Dict[str, Any]:
        """Get summary of all orders"""
        with self._lock:
            total_orders = len(self.orders)
            status_counts = {}
            total_volume = 0
            total_value = 0
            
            for order in self.orders.values():
                status_counts[order.status] = status_counts.get(order.status, 0) + 1
                
                if order.status == "FILLED":
                    total_volume += order.filled_quantity
                    if order.fill_price:
                        total_value += order.filled_quantity * order.fill_price
            
            return {
                "total_orders": total_orders,
                "status_counts": status_counts,
                "total_volume": total_volume,
                "total_value": total_value
            }
    
    def add_callback(self, callback: Callable):
        """Add order update callback"""
        self._callbacks.append(callback)
    
    def _on_order_update(self, order_id: int, status: str, 
                        filled: float, fill_price: float):
        """Handle order updates from connector"""
        try:
            with self._lock:
                if order_id in self.orders:
                    order = self.orders[order_id]
                    order.status = status
                    order.filled_quantity = int(filled)
                    order.remaining_quantity = order.quantity - order.filled_quantity
                    
                    if fill_price > 0:
                        order.fill_price = fill_price
            
            logger.info(f"Order {order_id} update: {status}, filled: {filled}")
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(order_id, status, filled, fill_price)
                except Exception as e:
                    logger.error(f"Error in order callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing order update: {e}")
    
    def cleanup_old_orders(self, max_age_hours: int = 24):
        """
        Clean up old completed orders
        
        Args:
            max_age_hours: Maximum age in hours to keep orders
        """
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            
            with self._lock:
                to_remove = []
                for order_id, order in self.orders.items():
                    if (order.status in ["FILLED", "CANCELLED", "REJECTED"] and 
                        order.timestamp < cutoff_time):
                        to_remove.append(order_id)
                
                for order_id in to_remove:
                    del self.orders[order_id]
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old orders")
                
        except Exception as e:
            logger.error(f"Error cleaning up old orders: {e}") 