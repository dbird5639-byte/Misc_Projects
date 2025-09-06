"""
Interactive Brokers API Connector

Handles connection to Interactive Brokers TWS/Gateway and manages
API communication for trading operations.
"""

import time
import threading
from typing import Dict, Any, Optional, Callable, List, Union

# Import IB API (would be installed via requirements.txt)
try:
    from loguru import logger  # type: ignore
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from ibapi.client import EClient  # type: ignore
    from ibapi.wrapper import EWrapper  # type: ignore
    from ibapi.contract import Contract  # type: ignore
    from ibapi.order import Order  # type: ignore
    from ibapi.common import TickerId, OrderId  # type: ignore
    IB_AVAILABLE = True
except ImportError:
    logger.warning("IB API not available - using mock implementation")
    IB_AVAILABLE = False
    # Create mock classes for when IB API is not available
    class EClient:
        def __init__(self, wrapper):
            pass
        def connect(self, host, port, client_id):
            pass
        def disconnect(self):
            pass
        def run(self):
            pass
        def reqMktData(self, reqId, contract, genericTickList, snapshot, regulatorySnapshot, mktDataOptions):
            pass
        def placeOrder(self, orderId, contract, order):
            pass
        def cancelOrder(self, orderId):
            pass
    
    class EWrapper:
        pass
    
    class Contract:
        def __init__(self):
            self.symbol = ""
            self.secType = ""
            self.exchange = ""
            self.currency = ""
    
    class Order:
        def __init__(self):
            self.action = ""
            self.totalQuantity = 0
            self.orderType = ""
            self.lmtPrice = 0.0
    
    TickerId = int
    OrderId = int

class IBConnector:
    """
    Interactive Brokers API connector
    
    Manages connection to TWS/Gateway and handles all API communication.
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, 
                 client_id: int = 1):
        """
        Initialize IB connector
        
        Args:
            host: TWS/Gateway host address
            port: TWS/Gateway port (7497 for paper, 7496 for live)
            client_id: Unique client identifier
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False
        self.next_order_id = 1
        self.orders: Dict[int, Dict[str, Any]] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.market_data: Dict[str, Dict[str, Any]] = {}
        
        # IB API client (if available)
        self._client: Optional[EClient] = None
        if IB_AVAILABLE:
            self._client = EClient(self)
        
        # Callbacks
        self.connection_callback: Optional[Callable] = None
        self.order_callback: Optional[Callable] = None
        self.position_callback: Optional[Callable] = None
        self.market_data_callback: Optional[Callable] = None
        
        # Threading
        self._lock = threading.Lock()
        self._message_thread: Optional[threading.Thread] = None
    
    def connect(self) -> bool:
        """
        Connect to Interactive Brokers TWS/Gateway
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to IB at {self.host}:{self.port}")
            
            if not IB_AVAILABLE:
                logger.warning("Using mock connection (IB API not available)")
                self.connected = True
                return True
            
            # Connect to TWS/Gateway
            if self._client:
                self._client.connect(self.host, self.port, self.client_id)
            
            # Start message processing thread
            self._message_thread = threading.Thread(target=self._process_messages)
            self._message_thread.daemon = True
            self._message_thread.start()
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.connected:
                logger.info("Successfully connected to Interactive Brokers")
                if self.connection_callback:
                    self.connection_callback(True)
                return True
            else:
                logger.error("Failed to connect to Interactive Brokers")
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Interactive Brokers"""
        try:
            logger.info("Disconnecting from Interactive Brokers")
            
            self.connected = False
            
            if IB_AVAILABLE and self._client:
                self._client.disconnect()
            
            # Stop message thread
            if self._message_thread and self._message_thread.is_alive():
                self._message_thread.join(timeout=5)
            
            logger.info("Disconnected from Interactive Brokers")
            
        except Exception as e:
            logger.error(f"Disconnection error: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to Interactive Brokers"""
        return self.connected
    
    def _process_messages(self):
        """Process incoming messages from IB API"""
        try:
            while self.connected:
                if IB_AVAILABLE and self._client:
                    self._client.run()
                else:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            self.connected = False
    
    def subscribe_market_data(self, symbol: str):
        """
        Subscribe to market data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
        """
        try:
            logger.info(f"Subscribing to market data for {symbol}")
            
            if not self.connected:
                logger.error("Not connected to Interactive Brokers")
                return
            
            # Create contract
            contract = self._create_stock_contract(symbol)
            
            # Subscribe to market data
            if IB_AVAILABLE and self._client:
                self._client.reqMktData(self.next_order_id, contract, "", False, False, [])
                self.next_order_id += 1
            
            # Initialize market data storage
            self.market_data[symbol] = {
                "last_price": 0.0,
                "bid": 0.0,
                "ask": 0.0,
                "volume": 0,
                "timestamp": time.time()
            }
            
            logger.info(f"Subscribed to market data for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to market data for {symbol}: {e}")
    
    def _create_stock_contract(self, symbol: str) -> Contract:
        """Create a stock contract for the given symbol"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract
    
    def place_order(self, symbol: str, action: str, quantity: int, 
                   order_type: str = "MKT", price: float = 0.0) -> int:
        """
        Place an order
        
        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares
            order_type: Order type (MKT, LMT, STP, etc.)
            price: Limit price (for LMT orders)
            
        Returns:
            Order ID
        """
        try:
            logger.info(f"Placing {order_type} {action} order for {quantity} shares of {symbol}")
            
            if not self.connected:
                logger.error("Not connected to Interactive Brokers")
                return -1
            
            # Create contract and order
            contract = self._create_stock_contract(symbol)
            order = self._create_order(action, quantity, order_type, price)
            
            # Store order information
            order_id = self.next_order_id
            self.orders[order_id] = {
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "order_type": order_type,
                "price": price,
                "status": "SUBMITTED",
                "timestamp": time.time()
            }
            
            # Place order
            if IB_AVAILABLE and self._client:
                self._client.placeOrder(order_id, contract, order)
                self.next_order_id += 1
            else:
                # Mock order placement
                logger.info(f"Mock order placed: {order_id}")
                self._mock_order_fill(order_id)
            
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return -1
    
    def _create_order(self, action: str, quantity: int, 
                     order_type: str, price: float) -> Order:
        """Create an order object"""
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = order_type
        if price > 0:
            order.lmtPrice = price
        return order
    
    def _mock_order_fill(self, order_id: int):
        """Simulate order fill for mock implementation"""
        def fill_order():
            time.sleep(1)  # Simulate processing time
            with self._lock:
                if order_id in self.orders:
                    self.orders[order_id]["status"] = "FILLED"
                    self.orders[order_id]["fill_price"] = 150.0  # Mock price
                    logger.info(f"Mock order {order_id} filled")
        
        threading.Thread(target=fill_order, daemon=True).start()
    
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
            
            if not self.connected:
                logger.error("Not connected to Interactive Brokers")
                return False
            
            if IB_AVAILABLE and self._client:
                self._client.cancelOrder(order_id)
            
            # Update order status
            if order_id in self.orders:
                self.orders[order_id]["status"] = "CANCELLED"
            
            logger.info(f"Order {order_id} cancelled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_orders(self) -> Dict[int, Dict[str, Any]]:
        """Get all orders"""
        with self._lock:
            return self.orders.copy()
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions"""
        with self._lock:
            return self.positions.copy()
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market data for a symbol"""
        with self._lock:
            return self.market_data.get(symbol)
    
    # IB API Callbacks (EWrapper methods)
    
    def connectAck(self):
        """Called when connection is established"""
        logger.info("IB connection acknowledged")
        self.connected = True
    
    def connectionClosed(self):
        """Called when connection is closed"""
        logger.warning("IB connection closed")
        self.connected = False
    
    def nextValidId(self, orderId: int):
        """Called with next valid order ID"""
        self.next_order_id = orderId
        logger.info(f"Next valid order ID: {orderId}")
    
    def orderStatus(self, orderId: int, status: str, filled: float,
                   remaining: float, avgFillPrice: float, permId: int,
                   parentId: int, lastFillPrice: float, clientId: int,
                   whyHeld: str, mktCapPrice: float):
        """Called when order status changes"""
        with self._lock:
            if orderId in self.orders:
                self.orders[orderId]["status"] = status
                self.orders[orderId]["filled"] = filled
                self.orders[orderId]["remaining"] = remaining
                self.orders[orderId]["avg_fill_price"] = avgFillPrice
        
        logger.info(f"Order {orderId} status: {status}")
        
        if self.order_callback:
            self.order_callback(orderId, status, filled, avgFillPrice)
    
    def position(self, account: str, contract: Contract, position: float,
                avgCost: float):
        """Called when position information is received"""
        symbol = contract.symbol
        
        with self._lock:
            self.positions[symbol] = {
                "account": account,
                "position": position,
                "avg_cost": avgCost,
                "contract": contract
            }
        
        logger.info(f"Position update: {symbol} = {position} shares")
        
        if self.position_callback:
            self.position_callback(symbol, position, avgCost)
    
    def tickPrice(self, reqId: int, tickType: int, price: float,
                  attrib: int):
        """Called when price data is received"""
        # Find symbol by reqId (simplified)
        symbol = f"SYMBOL_{reqId}"  # In real implementation, maintain reqId->symbol mapping
        
        with self._lock:
            if symbol in self.market_data:
                if tickType == 1:  # Bid
                    self.market_data[symbol]["bid"] = price
                elif tickType == 2:  # Ask
                    self.market_data[symbol]["ask"] = price
                elif tickType == 4:  # Last
                    self.market_data[symbol]["last_price"] = price
                
                self.market_data[symbol]["timestamp"] = time.time()
        
        if self.market_data_callback:
            self.market_data_callback(symbol, tickType, price)
    
    def error(self, reqId: int, errorCode: int, errorString: str):
        """Called when an error occurs"""
        logger.error(f"IB Error {errorCode}: {errorString} (reqId: {reqId})") 