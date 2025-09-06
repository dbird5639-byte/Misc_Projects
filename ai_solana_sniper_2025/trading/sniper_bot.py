"""
Sniper Bot for executing trades on Solana
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

from solana.rpc.async_api import AsyncClient
from solana.transaction import Transaction
from solana.keypair import Keypair
from solana.publickey import PublicKey
from solders.system_program import TransferParams, transfer
from solders.instruction import Instruction

from ..utils.logger import get_logger
from ..utils.notifications import NotificationManager

logger = get_logger(__name__)


@dataclass
class TradeOrder:
    """Represents a trade order"""
    order_id: str
    token_address: str
    action: str  # "buy" or "sell"
    amount: float
    price: float
    max_slippage: float
    status: str  # "pending", "executing", "completed", "failed", "cancelled"
    created_at: datetime
    executed_at: Optional[datetime] = None
    transaction_hash: Optional[str] = None
    actual_price: Optional[float] = None
    slippage: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class Position:
    """Represents a trading position"""
    token_address: str
    amount: float
    average_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime
    last_updated: datetime


class SniperBot:
    """
    Bot for executing trades on Solana
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.auto_trade = config.get("auto_trade", False)
        self.max_position_size = config.get("max_position_size", 0.05)
        self.max_slippage = config.get("max_slippage", 0.05)
        self.gas_limit = config.get("gas_limit", 300000)
        
        # Solana connection
        self.rpc_url = config.get("rpc_url", "https://api.mainnet-beta.solana.com")
        self.client = None
        
        # Wallet configuration
        self.wallet_private_key = config.get("wallet_private_key")
        self.wallet_keypair = None
        
        # Jupiter API for swaps
        self.jupiter_api_url = "https://quote-api.jup.ag/v6"
        self.jupiter_api_key = config.get("jupiter_api_key")
        
        # Trading state
        self.orders: Dict[str, TradeOrder] = {}
        self.positions: Dict[str, Position] = {}
        self.is_initialized = False
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_volume = 0.0
        self.total_fees = 0.0
        
        # Notifications
        self.notifications = None
        
    async def initialize(self):
        """Initialize the sniper bot"""
        try:
            # Initialize Solana client
            self.client = AsyncClient(self.rpc_url)
            
            # Initialize wallet
            if self.wallet_private_key:
                self.wallet_keypair = Keypair.from_secret_key(
                    bytes.fromhex(self.wallet_private_key)
                )
                logger.info(f"Wallet initialized: {self.wallet_keypair.public_key}")
            else:
                logger.warning("No wallet private key provided - trading disabled")
                self.enabled = False
            
            # Test connection
            await self._test_connection()
            
            self.is_initialized = True
            logger.info("Sniper bot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize sniper bot: {e}")
            return False
    
    async def _test_connection(self):
        """Test Solana connection"""
        if not self.client:
            logger.error("Solana client not initialized")
            raise Exception("Solana client not initialized")
            
        try:
            response = await self.client.get_slot()
            if response.value:
                logger.info("Solana connection successful")
            else:
                logger.warning("Solana connection test failed")
        except Exception as e:
            logger.error(f"Solana connection test failed: {e}")
            raise
    
    async def execute_buy_order(self, token_address: str, amount: float, 
                              max_price: Optional[float] = None) -> bool:
        """Execute a buy order"""
        if not self.enabled or not self.wallet_keypair:
            logger.warning("Trading disabled or wallet not initialized")
            return False
        
        try:
            logger.info(f"Executing buy order: {token_address}, amount: {amount}")
            
            # Create order
            order_id = f"buy_{int(time.time())}_{len(self.orders)}"
            order = TradeOrder(
                order_id=order_id,
                token_address=token_address,
                action="buy",
                amount=amount,
                price=max_price or 0.0,
                max_slippage=self.max_slippage,
                status="pending",
                created_at=datetime.now()
            )
            
            self.orders[order_id] = order
            
            # Get current price
            current_price = await self._get_token_price(token_address)
            if not current_price:
                order.status = "failed"
                order.error_message = "Could not get token price"
                return False
            
            # Check if price is acceptable
            if max_price and current_price > max_price:
                order.status = "failed"
                order.error_message = f"Price {current_price} exceeds max price {max_price}"
                return False
            
            # Execute the trade
            success = await self._execute_swap(
                order, "buy", token_address, amount, current_price
            )
            
            if success:
                self.successful_trades += 1
                self.total_volume += amount * current_price
                
                # Update position
                await self._update_position(token_address, amount, current_price, "buy")
                
                # Send notification
                if self.notifications and hasattr(self.notifications, 'send_notification'):
                    try:
                        await self.notifications.send_notification(
                            "💰 Buy Order Executed",
                            f"Token: {token_address}\nAmount: {amount}\nPrice: {current_price}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to send notification: {e}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing buy order: {e}")
            self.failed_trades += 1
            return False
    
    async def execute_sell_order(self, token_address: str, amount: float) -> bool:
        """Execute a sell order"""
        if not self.enabled or not self.wallet_keypair:
            logger.warning("Trading disabled or wallet not initialized")
            return False
        
        try:
            logger.info(f"Executing sell order: {token_address}, amount: {amount}")
            
            # Check if we have enough tokens
            position = self.positions.get(token_address)
            if not position or position.amount < amount:
                logger.warning(f"Insufficient tokens for sell: {token_address}")
                return False
            
            # Create order
            order_id = f"sell_{int(time.time())}_{len(self.orders)}"
            order = TradeOrder(
                order_id=order_id,
                token_address=token_address,
                action="sell",
                amount=amount,
                price=0.0,  # Market sell
                max_slippage=self.max_slippage,
                status="pending",
                created_at=datetime.now()
            )
            
            self.orders[order_id] = order
            
            # Get current price
            current_price = await self._get_token_price(token_address)
            if not current_price:
                order.status = "failed"
                order.error_message = "Could not get token price"
                return False
            
            # Execute the trade
            success = await self._execute_swap(
                order, "sell", token_address, amount, current_price
            )
            
            if success:
                self.successful_trades += 1
                self.total_volume += amount * current_price
                
                # Update position
                await self._update_position(token_address, amount, current_price, "sell")
                
                # Calculate P&L
                pnl = (current_price - position.average_price) * amount
                
                # Send notification
                if self.notifications and hasattr(self.notifications, 'send_notification'):
                    try:
                        await self.notifications.send_notification(
                            "💸 Sell Order Executed",
                            f"Token: {token_address}\nAmount: {amount}\nPrice: {current_price}\nP&L: {pnl}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to send notification: {e}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing sell order: {e}")
            self.failed_trades += 1
            return False
    
    async def _execute_swap(self, order: TradeOrder, action: str, token_address: str, 
                          amount: float, price: float) -> bool:
        """Execute a swap using Jupiter API"""
        try:
            order.status = "executing"
            
            # Get quote from Jupiter
            quote = await self._get_jupiter_quote(
                token_address, amount, action
            )
            
            if not quote:
                order.status = "failed"
                order.error_message = "Could not get swap quote"
                return False
            
            # Check slippage
            slippage = abs(quote["price"] - price) / price
            if slippage > self.max_slippage:
                order.status = "failed"
                order.error_message = f"Slippage {slippage:.2%} exceeds max {self.max_slippage:.2%}"
                return False
            
            # Execute swap
            transaction_hash = await self._execute_jupiter_swap(quote)
            
            if transaction_hash:
                order.status = "completed"
                order.executed_at = datetime.now()
                order.transaction_hash = transaction_hash
                order.actual_price = quote["price"]
                order.slippage = slippage
                
                self.total_trades += 1
                logger.info(f"Swap executed successfully: {transaction_hash}")
                return True
            else:
                order.status = "failed"
                order.error_message = "Swap execution failed"
                return False
                
        except Exception as e:
            order.status = "failed"
            order.error_message = str(e)
            logger.error(f"Error executing swap: {e}")
            return False
    
    async def _get_jupiter_quote(self, token_address: str, amount: float, action: str) -> Optional[Dict[str, Any]]:
        """Get swap quote from Jupiter"""
        try:
            # Determine input and output tokens
            if action == "buy":
                input_mint = "So11111111111111111111111111111111111111112"  # SOL
                output_mint = token_address
                input_amount = int(amount * 1e9)  # Convert to lamports
            else:
                input_mint = token_address
                output_mint = "So11111111111111111111111111111111111111112"  # SOL
                input_amount = int(amount * 1e6)  # Convert to token units (assuming 6 decimals)
            
            url = f"{self.jupiter_api_url}/quote"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(input_amount),
                "slippageBps": int(self.max_slippage * 10000)
            }
            
            headers = {}
            if self.jupiter_api_key:
                headers["Authorization"] = f"Bearer {self.jupiter_api_key}"
            
            async with self.client._provider.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "price": float(data.get("outAmount", 0)) / float(data.get("inAmount", 1)),
                        "inAmount": data.get("inAmount"),
                        "outAmount": data.get("outAmount"),
                        "swapTransaction": data.get("swapTransaction")
                    }
                else:
                    logger.warning(f"Jupiter quote API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting Jupiter quote: {e}")
            return None
    
    async def _execute_jupiter_swap(self, quote: Dict[str, Any]) -> Optional[str]:
        """Execute swap transaction"""
        try:
            # Decode and sign transaction
            swap_transaction = quote.get("swapTransaction")
            if not swap_transaction:
                return None
            
            # Decode transaction
            transaction_data = bytes.fromhex(swap_transaction)
            transaction = Transaction.deserialize(transaction_data)
            
            # Sign transaction
            transaction.sign(self.wallet_keypair)
            
            # Send transaction
            response = await self.client.send_transaction(transaction)
            
            if response.value:
                return response.value
            else:
                logger.error(f"Transaction failed: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing Jupiter swap: {e}")
            return None
    
    async def _get_token_price(self, token_address: str) -> Optional[float]:
        """Get current token price"""
        try:
            # Use Jupiter price API
            url = f"{self.jupiter_api_url}/price"
            params = {"ids": token_address}
            
            headers = {}
            if self.jupiter_api_key:
                headers["Authorization"] = f"Bearer {self.jupiter_api_key}"
            
            async with self.client._provider.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", {}).get(token_address, {}).get("price")
                else:
                    logger.warning(f"Jupiter price API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting token price: {e}")
            return None
    
    async def _update_position(self, token_address: str, amount: float, price: float, action: str):
        """Update trading position"""
        if action == "buy":
            if token_address in self.positions:
                # Update existing position
                position = self.positions[token_address]
                total_amount = position.amount + amount
                total_value = (position.amount * position.average_price) + (amount * price)
                position.average_price = total_value / total_amount
                position.amount = total_amount
                position.current_price = price
                position.last_updated = datetime.now()
            else:
                # Create new position
                self.positions[token_address] = Position(
                    token_address=token_address,
                    amount=amount,
                    average_price=price,
                    current_price=price,
                    unrealized_pnl=0.0,
                    entry_time=datetime.now(),
                    last_updated=datetime.now()
                )
        elif action == "sell":
            if token_address in self.positions:
                position = self.positions[token_address]
                position.amount -= amount
                position.current_price = price
                position.last_updated = datetime.now()
                
                # Remove position if amount is 0
                if position.amount <= 0:
                    del self.positions[token_address]
    
    async def get_balance(self) -> Dict[str, float]:
        """Get wallet balance"""
        try:
            if not self.client or not self.wallet_keypair:
                return {}
            
            response = await self.client.get_balance(self.wallet_keypair.public_key)
            if response.value:
                sol_balance = response.value / 1e9  # Convert lamports to SOL
                return {"SOL": sol_balance}
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {}
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        # Update unrealized P&L
        for token_address, position in self.positions.items():
            current_price = await self._get_token_price(token_address)
            if current_price:
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.average_price) * position.amount
                position.last_updated = datetime.now()
        
        return self.positions.copy()
    
    async def get_order_history(self) -> List[TradeOrder]:
        """Get order history"""
        return list(self.orders.values())
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        total_trades = self.successful_trades + self.failed_trades
        success_rate = self.successful_trades / max(total_trades, 1)
        
        return {
            "total_trades": total_trades,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "success_rate": success_rate,
            "total_volume": self.total_volume,
            "total_fees": self.total_fees,
            "active_positions": len(self.positions),
            "pending_orders": len([o for o in self.orders.values() if o.status == "pending"])
        }
    
    async def close(self):
        """Close the sniper bot"""
        if self.client:
            await self.client.close()
        
        logger.info("Sniper bot closed") 