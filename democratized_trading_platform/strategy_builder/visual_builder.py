"""
Visual Strategy Builder for Democratized Trading Platform
Allows users to build trading strategies using a drag-and-drop interface.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..backtesting.engine import BacktestEngine, BacktestConfig


class ComponentType(Enum):
    """Types of strategy components."""
    INDICATOR = "indicator"
    CONDITION = "condition"
    ACTION = "action"
    LOGIC = "logic"


class IndicatorType(Enum):
    """Available technical indicators."""
    SMA = "simple_moving_average"
    EMA = "exponential_moving_average"
    RSI = "relative_strength_index"
    MACD = "macd"
    BOLLINGER_BANDS = "bollinger_bands"
    STOCHASTIC = "stochastic"
    ATR = "average_true_range"
    VOLUME = "volume"
    PRICE = "price"


class ConditionType(Enum):
    """Types of conditions."""
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    EQUAL = "equal"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    BETWEEN = "between"
    OUTSIDE = "outside"


class ActionType(Enum):
    """Types of actions."""
    BUY = "buy"
    SELL = "sell"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    SET_STOP_LOSS = "set_stop_loss"
    SET_TAKE_PROFIT = "set_take_profit"


@dataclass
class Component:
    """Base component for strategy building."""
    id: str
    name: str
    component_type: ComponentType
    parameters: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)  # x, y coordinates
    connections: List[str] = field(default_factory=list)  # connected component IDs


@dataclass
class Indicator(Component):
    """Technical indicator component."""
    indicator_type: IndicatorType
    output_name: str = ""


@dataclass
class Condition(Component):
    """Condition component."""
    condition_type: ConditionType
    left_input: str = ""
    right_input: str = ""
    threshold: float = 0.0


@dataclass
class Action(Component):
    """Action component."""
    action_type: ActionType
    quantity: Union[float, str] = 1.0  # Can be fixed or dynamic
    conditions: List[str] = field(default_factory=list)  # Required conditions


@dataclass
class Logic(Component):
    """Logic component (AND, OR, NOT)."""
    logic_type: str = "AND"  # AND, OR, NOT
    inputs: List[str] = field(default_factory=list)


class VisualStrategyBuilder:
    """
    Visual strategy builder that allows users to create trading strategies
    without writing code.
    """
    
    def __init__(self):
        self.components = {}
        self.strategy_name = "My Strategy"
        self.description = ""
        self.logger = logging.getLogger(__name__)
        
        # Initialize with common components
        self._initialize_common_components()
    
    def _initialize_common_components(self):
        """Initialize commonly used components."""
        # Price indicators
        self.add_component(Indicator(
            id="price_close",
            name="Close Price",
            component_type=ComponentType.INDICATOR,
            indicator_type=IndicatorType.PRICE,
            output_name="close"
        ))
        
        self.add_component(Indicator(
            id="price_open",
            name="Open Price",
            component_type=ComponentType.INDICATOR,
            indicator_type=IndicatorType.PRICE,
            output_name="open"
        ))
        
        # Moving averages
        self.add_component(Indicator(
            id="sma_20",
            name="SMA 20",
            component_type=ComponentType.INDICATOR,
            indicator_type=IndicatorType.SMA,
            parameters={"period": 20},
            output_name="sma_20"
        ))
        
        self.add_component(Indicator(
            id="ema_12",
            name="EMA 12",
            component_type=ComponentType.INDICATOR,
            indicator_type=IndicatorType.EMA,
            parameters={"period": 12},
            output_name="ema_12"
        ))
        
        # RSI
        self.add_component(Indicator(
            id="rsi_14",
            name="RSI 14",
            component_type=ComponentType.INDICATOR,
            indicator_type=IndicatorType.RSI,
            parameters={"period": 14},
            output_name="rsi_14"
        ))
    
    def add_component(self, component: Component) -> str:
        """Add a component to the strategy."""
        self.components[component.id] = component
        self.logger.info(f"Added component: {component.name} ({component.id})")
        return component.id
    
    def remove_component(self, component_id: str) -> bool:
        """Remove a component from the strategy."""
        if component_id in self.components:
            # Remove connections to this component
            for comp in self.components.values():
                if component_id in comp.connections:
                    comp.connections.remove(component_id)
            
            # Remove the component
            del self.components[component_id]
            self.logger.info(f"Removed component: {component_id}")
            return True
        return False
    
    def connect_components(self, from_id: str, to_id: str) -> bool:
        """Connect two components."""
        if from_id in self.components and to_id in self.components:
            if to_id not in self.components[from_id].connections:
                self.components[from_id].connections.append(to_id)
                self.logger.info(f"Connected {from_id} to {to_id}")
                return True
        return False
    
    def disconnect_components(self, from_id: str, to_id: str) -> bool:
        """Disconnect two components."""
        if from_id in self.components and to_id in self.components[from_id].connections:
            self.components[from_id].connections.remove(to_id)
            self.logger.info(f"Disconnected {from_id} from {to_id}")
            return True
        return False
    
    def update_component_parameters(self, component_id: str, parameters: Dict[str, Any]) -> bool:
        """Update component parameters."""
        if component_id in self.components:
            self.components[component_id].parameters.update(parameters)
            self.logger.info(f"Updated parameters for {component_id}")
            return True
        return False
    
    def get_component(self, component_id: str) -> Optional[Component]:
        """Get a component by ID."""
        return self.components.get(component_id)
    
    def get_all_components(self) -> Dict[str, Component]:
        """Get all components."""
        return self.components.copy()
    
    def validate_strategy(self) -> Dict[str, Any]:
        """Validate the strategy for completeness."""
        errors = []
        warnings = []
        
        # Check for required components
        has_indicators = any(c.component_type == ComponentType.INDICATOR for c in self.components.values())
        has_conditions = any(c.component_type == ComponentType.CONDITION for c in self.components.values())
        has_actions = any(c.component_type == ComponentType.ACTION for c in self.components.values())
        
        if not has_indicators:
            errors.append("Strategy must have at least one indicator")
        
        if not has_conditions:
            errors.append("Strategy must have at least one condition")
        
        if not has_actions:
            errors.append("Strategy must have at least one action")
        
        # Check for disconnected components
        connected_components = set()
        for component in self.components.values():
            connected_components.update(component.connections)
            connected_components.add(component.id)
        
        for component_id in self.components:
            if component_id not in connected_components:
                warnings.append(f"Component {component_id} is not connected to the strategy")
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            errors.append("Strategy has circular dependencies")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "component_count": len(self.components),
            "has_indicators": has_indicators,
            "has_conditions": has_conditions,
            "has_actions": has_actions
        }
    
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies in the strategy."""
        visited = set()
        rec_stack = set()
        
        def dfs(component_id):
            visited.add(component_id)
            rec_stack.add(component_id)
            
            component = self.components.get(component_id)
            if component:
                for connected_id in component.connections:
                    if connected_id not in visited:
                        if dfs(connected_id):
                            return True
                    elif connected_id in rec_stack:
                        return True
            
            rec_stack.remove(component_id)
            return False
        
        for component_id in self.components:
            if component_id not in visited:
                if dfs(component_id):
                    return True
        
        return False
    
    def generate_python_code(self) -> str:
        """Generate Python code from the visual strategy."""
        validation = self.validate_strategy()
        if not validation["is_valid"]:
            raise ValueError(f"Strategy is not valid: {validation['errors']}")
        
        code_lines = [
            "import pandas as pd",
            "import numpy as np",
            "",
            f"def {self.strategy_name.lower().replace(' ', '_')}_strategy(data, params=None):",
            f'    """',
            f'    Generated strategy: {self.strategy_name}',
            f'    {self.description}',
            f'    """',
            "    data = data.copy()",
            "    signals = pd.Series(0, index=data.index)",
            ""
        ]
        
        # Generate indicator calculations
        indicators = [c for c in self.components.values() if c.component_type == ComponentType.INDICATOR]
        for indicator in indicators:
            code_lines.extend(self._generate_indicator_code(indicator))
        
        # Generate condition and action logic
        code_lines.extend(self._generate_logic_code())
        
        code_lines.extend([
            "    return signals",
            ""
        ])
        
        return "\n".join(code_lines)
    
    def _generate_indicator_code(self, indicator: Indicator) -> List[str]:
        """Generate code for an indicator."""
        code_lines = []
        
        if indicator.indicator_type == IndicatorType.SMA:
            period = indicator.parameters.get("period", 20)
            code_lines.extend([
                f"    # {indicator.name}",
                f"    data['{indicator.output_name}'] = data['close'].rolling({period}).mean()",
                ""
            ])
        
        elif indicator.indicator_type == IndicatorType.EMA:
            period = indicator.parameters.get("period", 12)
            code_lines.extend([
                f"    # {indicator.name}",
                f"    data['{indicator.output_name}'] = data['close'].ewm(span={period}).mean()",
                ""
            ])
        
        elif indicator.indicator_type == IndicatorType.RSI:
            period = indicator.parameters.get("period", 14)
            code_lines.extend([
                f"    # {indicator.name}",
                f"    delta = data['close'].diff()",
                f"    gain = (delta.where(delta > 0, 0)).rolling(window={period}).mean()",
                f"    loss = (-delta.where(delta < 0, 0)).rolling(window={period}).mean()",
                f"    rs = gain / loss",
                f"    data['{indicator.output_name}'] = 100 - (100 / (1 + rs))",
                ""
            ])
        
        elif indicator.indicator_type == IndicatorType.MACD:
            fast = indicator.parameters.get("fast", 12)
            slow = indicator.parameters.get("slow", 26)
            signal = indicator.parameters.get("signal", 9)
            code_lines.extend([
                f"    # {indicator.name}",
                f"    ema_fast = data['close'].ewm(span={fast}).mean()",
                f"    ema_slow = data['close'].ewm(span={slow}).mean()",
                f"    data['{indicator.output_name}_macd'] = ema_fast - ema_slow",
                f"    data['{indicator.output_name}_signal'] = data['{indicator.output_name}_macd'].ewm(span={signal}).mean()",
                f"    data['{indicator.output_name}_histogram'] = data['{indicator.output_name}_macd'] - data['{indicator.output_name}_signal']",
                ""
            ])
        
        elif indicator.indicator_type == IndicatorType.BOLLINGER_BANDS:
            period = indicator.parameters.get("period", 20)
            std = indicator.parameters.get("std", 2)
            code_lines.extend([
                f"    # {indicator.name}",
                f"    sma = data['close'].rolling({period}).mean()",
                f"    std_dev = data['close'].rolling({period}).std()",
                f"    data['{indicator.output_name}_upper'] = sma + (std_dev * {std})",
                f"    data['{indicator.output_name}_lower'] = sma - (std_dev * {std})",
                f"    data['{indicator.output_name}_middle'] = sma",
                ""
            ])
        
        elif indicator.indicator_type == IndicatorType.PRICE:
            # Price is already available as 'close', 'open', etc.
            pass
        
        return code_lines
    
    def _generate_logic_code(self) -> List[str]:
        """Generate the main logic code for conditions and actions."""
        code_lines = []
        
        # Find action components
        actions = [c for c in self.components.values() if c.component_type == ComponentType.ACTION]
        
        for action in actions:
            code_lines.extend(self._generate_action_code(action))
        
        return code_lines
    
    def _generate_action_code(self, action: Action) -> List[str]:
        """Generate code for an action."""
        code_lines = []
        
        # Find conditions that lead to this action
        conditions = []
        for component in self.components.values():
            if action.id in component.connections:
                if component.component_type == ComponentType.CONDITION:
                    conditions.append(component)
                elif component.component_type == ComponentType.LOGIC:
                    # Handle logic components
                    conditions.extend(self._get_logic_conditions(component))
        
        if not conditions:
            return code_lines
        
        # Generate condition logic
        condition_code = self._generate_condition_expression(conditions)
        
        # Generate action code
        if action.action_type == ActionType.BUY:
            code_lines.extend([
                f"    # {action.name}",
                f"    buy_condition = {condition_code}",
                f"    signals[buy_condition] = 1",
                ""
            ])
        
        elif action.action_type == ActionType.SELL:
            code_lines.extend([
                f"    # {action.name}",
                f"    sell_condition = {condition_code}",
                f"    signals[sell_condition] = -1",
                ""
            ])
        
        return code_lines
    
    def _generate_condition_expression(self, conditions: List[Condition]) -> str:
        """Generate a boolean expression for conditions."""
        if len(conditions) == 1:
            return self._generate_single_condition(conditions[0])
        
        # Multiple conditions - assume AND logic for now
        expressions = [self._generate_single_condition(c) for c in conditions]
        return " & ".join(expressions)
    
    def _generate_single_condition(self, condition: Condition) -> str:
        """Generate code for a single condition."""
        if condition.condition_type == ConditionType.GREATER_THAN:
            return f"(data['{condition.left_input}'] > {condition.threshold})"
        
        elif condition.condition_type == ConditionType.LESS_THAN:
            return f"(data['{condition.left_input}'] < {condition.threshold})"
        
        elif condition.condition_type == ConditionType.CROSSES_ABOVE:
            return f"(data['{condition.left_input}'].shift(1) <= {condition.right_input}) & (data['{condition.left_input}'] > {condition.right_input})"
        
        elif condition.condition_type == ConditionType.CROSSES_BELOW:
            return f"(data['{condition.left_input}'].shift(1) >= {condition.right_input}) & (data['{condition.left_input}'] < {condition.right_input})"
        
        elif condition.condition_type == ConditionType.BETWEEN:
            return f"(data['{condition.left_input}'] >= {condition.threshold}) & (data['{condition.left_input}'] <= {condition.right_input})"
        
        return f"(data['{condition.left_input}'] == {condition.threshold})"
    
    def _get_logic_conditions(self, logic: Logic) -> List[Condition]:
        """Get conditions connected to a logic component."""
        conditions = []
        for input_id in logic.inputs:
            component = self.components.get(input_id)
            if component and component.component_type == ComponentType.CONDITION:
                conditions.append(component)
        return conditions
    
    def save_strategy(self, filename: str) -> bool:
        """Save strategy to a JSON file."""
        try:
            strategy_data = {
                "name": self.strategy_name,
                "description": self.description,
                "components": {}
            }
            
            for component_id, component in self.components.items():
                strategy_data["components"][component_id] = {
                    "id": component.id,
                    "name": component.name,
                    "component_type": component.component_type.value,
                    "parameters": component.parameters,
                    "position": component.position,
                    "connections": component.connections
                }
                
                # Add specific fields based on component type
                if isinstance(component, Indicator):
                    strategy_data["components"][component_id]["indicator_type"] = component.indicator_type.value
                    strategy_data["components"][component_id]["output_name"] = component.output_name
                
                elif isinstance(component, Condition):
                    strategy_data["components"][component_id]["condition_type"] = component.condition_type.value
                    strategy_data["components"][component_id]["left_input"] = component.left_input
                    strategy_data["components"][component_id]["right_input"] = component.right_input
                    strategy_data["components"][component_id]["threshold"] = component.threshold
                
                elif isinstance(component, Action):
                    strategy_data["components"][component_id]["action_type"] = component.action_type.value
                    strategy_data["components"][component_id]["quantity"] = component.quantity
                    strategy_data["components"][component_id]["conditions"] = component.conditions
                
                elif isinstance(component, Logic):
                    strategy_data["components"][component_id]["logic_type"] = component.logic_type
                    strategy_data["components"][component_id]["inputs"] = component.inputs
            
            with open(filename, 'w') as f:
                json.dump(strategy_data, f, indent=2)
            
            self.logger.info(f"Strategy saved to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving strategy: {e}")
            return False
    
    def load_strategy(self, filename: str) -> bool:
        """Load strategy from a JSON file."""
        try:
            with open(filename, 'r') as f:
                strategy_data = json.load(f)
            
            # Clear current strategy
            self.components.clear()
            
            # Load strategy metadata
            self.strategy_name = strategy_data.get("name", "Loaded Strategy")
            self.description = strategy_data.get("description", "")
            
            # Load components
            for component_id, comp_data in strategy_data["components"].items():
                component_type = ComponentType(comp_data["component_type"])
                
                if component_type == ComponentType.INDICATOR:
                    component = Indicator(
                        id=comp_data["id"],
                        name=comp_data["name"],
                        component_type=component_type,
                        parameters=comp_data.get("parameters", {}),
                        position=comp_data.get("position", {}),
                        connections=comp_data.get("connections", []),
                        indicator_type=IndicatorType(comp_data["indicator_type"]),
                        output_name=comp_data.get("output_name", "")
                    )
                
                elif component_type == ComponentType.CONDITION:
                    component = Condition(
                        id=comp_data["id"],
                        name=comp_data["name"],
                        component_type=component_type,
                        parameters=comp_data.get("parameters", {}),
                        position=comp_data.get("position", {}),
                        connections=comp_data.get("connections", []),
                        condition_type=ConditionType(comp_data["condition_type"]),
                        left_input=comp_data.get("left_input", ""),
                        right_input=comp_data.get("right_input", ""),
                        threshold=comp_data.get("threshold", 0.0)
                    )
                
                elif component_type == ComponentType.ACTION:
                    component = Action(
                        id=comp_data["id"],
                        name=comp_data["name"],
                        component_type=component_type,
                        parameters=comp_data.get("parameters", {}),
                        position=comp_data.get("position", {}),
                        connections=comp_data.get("connections", []),
                        action_type=ActionType(comp_data["action_type"]),
                        quantity=comp_data.get("quantity", 1.0),
                        conditions=comp_data.get("conditions", [])
                    )
                
                elif component_type == ComponentType.LOGIC:
                    component = Logic(
                        id=comp_data["id"],
                        name=comp_data["name"],
                        component_type=component_type,
                        parameters=comp_data.get("parameters", {}),
                        position=comp_data.get("position", {}),
                        connections=comp_data.get("connections", []),
                        logic_type=comp_data.get("logic_type", "AND"),
                        inputs=comp_data.get("inputs", [])
                    )
                
                self.components[component_id] = component
            
            self.logger.info(f"Strategy loaded from {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading strategy: {e}")
            return False
    
    def run_backtest(self, config: BacktestConfig) -> Any:
        """Run a backtest on the generated strategy."""
        try:
            # Generate Python code
            code = self.generate_python_code()
            
            # Create a temporary module with the strategy
            import types
            module = types.ModuleType("generated_strategy")
            exec(code, module.__dict__)
            
            # Get the strategy function
            strategy_func = getattr(module, f"{self.strategy_name.lower().replace(' ', '_')}_strategy")
            
            # Run backtest
            engine = BacktestEngine(config)
            result = engine.run_backtest(strategy_func)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Create a visual strategy builder
    builder = VisualStrategyBuilder()
    
    # Add components for a simple moving average crossover strategy
    sma_20 = builder.get_component("sma_20")
    sma_50 = builder.add_component(Indicator(
        id="sma_50",
        name="SMA 50",
        component_type=ComponentType.INDICATOR,
        indicator_type=IndicatorType.SMA,
        parameters={"period": 50},
        output_name="sma_50"
    ))
    
    # Add crossover condition
    crossover_condition = builder.add_component(Condition(
        id="crossover",
        name="SMA Crossover",
        component_type=ComponentType.CONDITION,
        condition_type=ConditionType.CROSSES_ABOVE,
        left_input="sma_20",
        right_input="sma_50"
    ))
    
    # Add buy action
    buy_action = builder.add_component(Action(
        id="buy",
        name="Buy Signal",
        component_type=ComponentType.ACTION,
        action_type=ActionType.BUY,
        quantity=1.0
    ))
    
    # Connect components
    builder.connect_components("sma_20", "crossover")
    builder.connect_components("sma_50", "crossover")
    builder.connect_components("crossover", "buy")
    
    # Validate strategy
    validation = builder.validate_strategy()
    print(f"Strategy validation: {validation}")
    
    # Generate Python code
    code = builder.generate_python_code()
    print("Generated Python code:")
    print(code)
    
    # Save strategy
    builder.save_strategy("sma_crossover_strategy.json") 