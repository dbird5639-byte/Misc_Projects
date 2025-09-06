"""
Straten-Inspired Strategy Generator
Inspired by Jacob Amaral's Straten tool for automated strategy generation

This module provides automated strategy creation from common indicators,
multi-language code generation, and template-based strategy development.
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


class Language(Enum):
    """Supported programming languages for code generation"""
    PYTHON = "python"
    CPP = "cpp"
    EASYLANGUAGE = "easylanguage"
    CSHARP = "csharp"
    NINJATRADER = "ninjatrader"


class StrategyType(Enum):
    """Types of trading strategies"""
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING = "swing"


@dataclass
class Indicator:
    """Trading indicator configuration"""
    name: str
    parameters: Dict[str, Any]
    description: str
    category: str


@dataclass
class Strategy:
    """Generated trading strategy"""
    name: str
    type: StrategyType
    indicators: List[Indicator]
    logic: str
    parameters: Dict[str, Any]
    code: Dict[str, str]  # Language -> code mapping
    description: str


class StratenGenerator:
    """
    Jacob Amaral's Straten-inspired strategy generator
    
    Automates the creation of trading strategies from common indicators,
    generates code for multiple platforms, and provides template-based
    strategy development.
    """
    
    def __init__(self):
        self.indicators = self._load_indicators()
        self.templates = self._load_templates()
        self.code_generators = self._load_code_generators()
    
    def _load_indicators(self) -> Dict[str, Indicator]:
        """Load available trading indicators"""
        return {
            "bollinger_bands": Indicator(
                name="Bollinger Bands",
                parameters={"period": 20, "std_dev": 2},
                description="Volatility indicator with upper/lower bands",
                category="volatility"
            ),
            "rsi": Indicator(
                name="Relative Strength Index",
                parameters={"period": 14},
                description="Momentum oscillator measuring speed of price changes",
                category="momentum"
            ),
            "macd": Indicator(
                name="MACD",
                parameters={"fast": 12, "slow": 26, "signal": 9},
                description="Trend-following momentum indicator",
                category="trend"
            ),
            "sma": Indicator(
                name="Simple Moving Average",
                parameters={"period": 50},
                description="Trend indicator showing average price over period",
                category="trend"
            ),
            "ema": Indicator(
                name="Exponential Moving Average",
                parameters={"period": 20},
                description="Weighted moving average giving more weight to recent prices",
                category="trend"
            ),
            "stochastic": Indicator(
                name="Stochastic Oscillator",
                parameters={"k_period": 14, "d_period": 3},
                description="Momentum indicator comparing closing price to price range",
                category="momentum"
            ),
            "atr": Indicator(
                name="Average True Range",
                parameters={"period": 14},
                description="Volatility indicator measuring market volatility",
                category="volatility"
            ),
            "adx": Indicator(
                name="Average Directional Index",
                parameters={"period": 14},
                description="Trend strength indicator",
                category="trend"
            ),
            "cci": Indicator(
                name="Commodity Channel Index",
                parameters={"period": 20},
                description="Momentum oscillator measuring price deviations",
                category="momentum"
            ),
            "williams_r": Indicator(
                name="Williams %R",
                parameters={"period": 14},
                description="Momentum oscillator measuring overbought/oversold levels",
                category="momentum"
            )
        }
    
    def _load_templates(self) -> Dict[str, Dict]:
        """Load strategy templates"""
        return {
            "mean_reversion": {
                "description": "Mean reversion strategies that trade against the trend",
                "indicators": ["bollinger_bands", "rsi", "stochastic"],
                "logic": "Buy when price is oversold, sell when overbought",
                "entry_rules": [
                    "Price touches lower Bollinger Band",
                    "RSI below 30",
                    "Stochastic below 20"
                ],
                "exit_rules": [
                    "Price touches upper Bollinger Band",
                    "RSI above 70",
                    "Stochastic above 80"
                ]
            },
            "trend_following": {
                "description": "Trend following strategies that ride the trend",
                "indicators": ["sma", "ema", "macd", "adx"],
                "logic": "Buy when trend is up, sell when trend is down",
                "entry_rules": [
                    "Price above moving average",
                    "MACD line above signal line",
                    "ADX above 25 indicating strong trend"
                ],
                "exit_rules": [
                    "Price crosses below moving average",
                    "MACD line crosses below signal line",
                    "ADX falls below 20"
                ]
            },
            "momentum": {
                "description": "Momentum strategies that follow price momentum",
                "indicators": ["rsi", "macd", "cci", "williams_r"],
                "logic": "Buy when momentum is strong, sell when momentum weakens",
                "entry_rules": [
                    "RSI above 50 and rising",
                    "MACD histogram increasing",
                    "CCI above 100"
                ],
                "exit_rules": [
                    "RSI crosses below 50",
                    "MACD histogram decreasing",
                    "CCI crosses below 100"
                ]
            },
            "breakout": {
                "description": "Breakout strategies that trade price breakouts",
                "indicators": ["bollinger_bands", "atr", "sma"],
                "logic": "Buy when price breaks above resistance, sell when breaks below support",
                "entry_rules": [
                    "Price breaks above upper Bollinger Band",
                    "Volume above average",
                    "ATR increasing"
                ],
                "exit_rules": [
                    "Price returns to Bollinger Band center",
                    "Volume decreasing",
                    "ATR decreasing"
                ]
            }
        }
    
    def _load_code_generators(self) -> Dict[str, callable]:
        """Load code generators for different languages"""
        return {
            Language.PYTHON: self._generate_python_code,
            Language.CPP: self._generate_cpp_code,
            Language.EASYLANGUAGE: self._generate_easylanguage_code,
            Language.CSHARP: self._generate_csharp_code,
            Language.NINJATRADER: self._generate_ninjatrader_code
        }
    
    def generate_strategy(
        self,
        indicators: List[str],
        logic: str,
        language: str = "python",
        strategy_name: Optional[str] = None,
        custom_parameters: Optional[Dict] = None
    ) -> Strategy:
        """
        Generate a trading strategy based on indicators and logic
        
        Args:
            indicators: List of indicator names to use
            logic: Strategy logic type (mean_reversion, trend_following, etc.)
            language: Target programming language
            strategy_name: Custom strategy name
            custom_parameters: Custom parameters for indicators
            
        Returns:
            Generated Strategy object
        """
        # Validate inputs
        if logic not in self.templates:
            raise ValueError(f"Unknown strategy logic: {logic}")
        
        # Get template
        template = self.templates[logic]
        
        # Validate indicators
        valid_indicators = []
        for indicator_name in indicators:
            if indicator_name in self.indicators:
                valid_indicators.append(self.indicators[indicator_name])
            else:
                print(f"Warning: Unknown indicator {indicator_name}")
        
        if not valid_indicators:
            raise ValueError("No valid indicators provided")
        
        # Generate strategy name
        if not strategy_name:
            strategy_name = f"{logic}_{'_'.join([ind.name.lower() for ind in valid_indicators])}"
        
        # Update parameters with custom values
        parameters = {}
        for indicator in valid_indicators:
            params = indicator.parameters.copy()
            if custom_parameters and indicator.name.lower() in custom_parameters:
                params.update(custom_parameters[indicator.name.lower()])
            parameters[indicator.name.lower()] = params
        
        # Generate code for all languages
        code = {}
        for lang in Language:
            try:
                code[lang.value] = self.code_generators[lang](
                    strategy_name, valid_indicators, template, parameters
                )
            except Exception as e:
                print(f"Warning: Failed to generate {lang.value} code: {e}")
        
        return Strategy(
            name=strategy_name,
            type=StrategyType(logic),
            indicators=valid_indicators,
            logic=template["logic"],
            parameters=parameters,
            code=code,
            description=template["description"]
        )
    
    def _generate_python_code(
        self,
        strategy_name: str,
        indicators: List[Indicator],
        template: Dict,
        parameters: Dict
    ) -> str:
        """Generate Python code for the strategy"""
        code = f'''"""
{strategy_name} - {template['description']}
Generated by Straten-Inspired Strategy Generator
"""

import pandas as pd
import numpy as np
import talib

class {strategy_name.replace('_', '').title()}Strategy:
    """{template['description']}"""
    
    def __init__(self):
        self.name = "{strategy_name}"
        self.parameters = {parameters}
        self.position = 0
        self.entry_price = 0
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = data.copy()
        
'''
        
        # Add indicator calculations
        for indicator in indicators:
            if indicator.name.lower() == "bollinger_bands":
                code += f'''        # {indicator.name}
        upper, middle, lower = talib.BBANDS(
            df['close'], 
            timeperiod={parameters['bollinger_bands']['period']}, 
            nbdevup={parameters['bollinger_bands']['std_dev']}, 
            nbdevdn={parameters['bollinger_bands']['std_dev']}
        )
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle
        
'''
            elif indicator.name.lower() == "rsi":
                code += f'''        # {indicator.name}
        df['rsi'] = talib.RSI(df['close'], timeperiod={parameters['rsi']['period']})
        
'''
            elif indicator.name.lower() == "macd":
                code += f'''        # {indicator.name}
        macd, signal, hist = talib.MACD(
            df['close'], 
            fastperiod={parameters['macd']['fast']}, 
            slowperiod={parameters['macd']['slow']}, 
            signalperiod={parameters['macd']['signal']}
        )
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
'''
            elif indicator.name.lower() == "sma":
                code += f'''        # {indicator.name}
        df['sma'] = talib.SMA(df['close'], timeperiod={parameters['sma']['period']})
        
'''
            elif indicator.name.lower() == "ema":
                code += f'''        # {indicator.name}
        df['ema'] = talib.EMA(df['close'], timeperiod={parameters['ema']['period']})
        
'''
        
        code += '''        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        df = self.calculate_indicators(data)
        df['signal'] = 0
        
'''
        
        # Add signal logic based on template
        if template.get("entry_rules"):
            code += "        # Entry rules\n"
            for rule in template["entry_rules"]:
                code += f"        # {rule}\n"
        
        if template.get("exit_rules"):
            code += "        # Exit rules\n"
            for rule in template["exit_rules"]:
                code += f"        # {rule}\n"
        
        code += '''        
        return df
    
    def execute_trade(self, signal: int, price: float):
        """Execute trade based on signal"""
        if signal == 1 and self.position <= 0:
            # Buy signal
            self.position = 1
            self.entry_price = price
            print(f"BUY at {price}")
        elif signal == -1 and self.position >= 0:
            # Sell signal
            self.position = -1
            self.entry_price = price
            print(f"SELL at {price}")
        elif signal == 0:
            # Close position
            if self.position != 0:
                pnl = (price - self.entry_price) * self.position
                print(f"CLOSE at {price}, PnL: {pnl:.2f}")
                self.position = 0
                self.entry_price = 0

# Usage example
if __name__ == "__main__":
    strategy = {strategy_name.replace('_', '').title()}Strategy()
    # Load your data and run strategy
    # data = pd.read_csv('your_data.csv')
    # signals = strategy.generate_signals(data)
'''
        
        return code
    
    def _generate_cpp_code(
        self,
        strategy_name: str,
        indicators: List[Indicator],
        template: Dict,
        parameters: Dict
    ) -> str:
        """Generate C++ code for NinjaTrader"""
        code = f'''// {strategy_name} - {template['description']}
// Generated by Straten-Inspired Strategy Generator
// For NinjaTrader 8

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{{
    public class {strategy_name.replace('_', '').title()}Strategy : Strategy
    {{
        private BB BB1;
        private RSI RSI1;
        private SMA SMA1;
        
        protected override void OnStateChange()
        {{
            if (State == State.SetDefaults)
            {{
                Description = "{template['description']}";
                Name = "{strategy_name}";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                IsFillLimitOnTouch = false;
                MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution = OrderFillResolution.Standard;
                Slippage = 0;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TimeInForce = TimeInForce.Gtc;
                TraceOrders = false;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade = 20;
            }}
            else if (State == State.Configure)
            {{
                // Add indicators
'''
        
        # Add indicator configurations
        for indicator in indicators:
            if indicator.name.lower() == "bollinger_bands":
                code += f'''                BB1 = BB({parameters['bollinger_bands']['period']}, {parameters['bollinger_bands']['std_dev']});
                AddChartIndicator(BB1);
'''
            elif indicator.name.lower() == "rsi":
                code += f'''                RSI1 = RSI({parameters['rsi']['period']}, 1);
                AddChartIndicator(RSI1);
'''
            elif indicator.name.lower() == "sma":
                code += f'''                SMA1 = SMA({parameters['sma']['period']});
                AddChartIndicator(SMA1);
'''
        
        code += '''            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < BarsRequiredToTrade) return;
            
            // Entry logic
'''
        
        # Add entry logic
        if template.get("entry_rules"):
            for rule in template["entry_rules"]:
                code += f"            // {rule}\n"
        
        code += '''            
            // Exit logic
'''
        
        # Add exit logic
        if template.get("exit_rules"):
            for rule in template["exit_rules"]:
                code += f"            // {rule}\n"
        
        code += '''        }
    }
}'''
        
        return code
    
    def _generate_easylanguage_code(
        self,
        strategy_name: str,
        indicators: List[Indicator],
        template: Dict,
        parameters: Dict
    ) -> str:
        """Generate EasyLanguage code for TradeStation"""
        code = f'''// {strategy_name} - {template['description']}
// Generated by Straten-Inspired Strategy Generator
// For TradeStation

inputs:
'''
        
        # Add inputs for parameters
        for indicator in indicators:
            if indicator.name.lower() == "bollinger_bands":
                code += f'''    BBPeriod({parameters['bollinger_bands']['period']}),
    BBStdDev({parameters['bollinger_bands']['std_dev']}),
'''
            elif indicator.name.lower() == "rsi":
                code += f'''    RSIPeriod({parameters['rsi']['period']}),
'''
            elif indicator.name.lower() == "sma":
                code += f'''    SMAPeriod({parameters['sma']['period']}),
'''
        
        code += '''    StopLoss(100),
    TakeProfit(200);

variables:
'''
        
        # Add variables for indicators
        for indicator in indicators:
            if indicator.name.lower() == "bollinger_bands":
                code += '''    BBUpper(0),
    BBLower(0),
    BBMiddle(0),
'''
            elif indicator.name.lower() == "rsi":
                code += '''    RSIValue(0),
'''
            elif indicator.name.lower() == "sma":
                code += '''    SMAValue(0),
'''
        
        code += '''    EntryPrice(0);

// Calculate indicators
'''
        
        # Add indicator calculations
        for indicator in indicators:
            if indicator.name.lower() == "bollinger_bands":
                code += f'''BBUpper = Average(Close, BBPeriod) + BBStdDev * StdDev(Close, BBPeriod);
BBLower = Average(Close, BBPeriod) - BBStdDev * StdDev(Close, BBPeriod);
BBMiddle = Average(Close, BBPeriod);
'''
            elif indicator.name.lower() == "rsi":
                code += f'''RSIValue = RSI(Close, RSIPeriod);
'''
            elif indicator.name.lower() == "sma":
                code += f'''SMAValue = Average(Close, SMAPeriod);
'''
        
        code += '''
// Entry and exit logic
'''
        
        # Add entry/exit logic
        if template.get("entry_rules"):
            code += "// Entry rules\n"
            for rule in template["entry_rules"]:
                code += f"// {rule}\n"
        
        if template.get("exit_rules"):
            code += "// Exit rules\n"
            for rule in template["exit_rules"]:
                code += f"// {rule}\n"
        
        code += '''
if MarketPosition = 0 then begin
    // Entry logic here
end;

if MarketPosition = 1 then begin
    // Exit logic here
end;
'''
        
        return code
    
    def _generate_csharp_code(
        self,
        strategy_name: str,
        indicators: List[Indicator],
        template: Dict,
        parameters: Dict
    ) -> str:
        """Generate C# code for general use"""
        # Similar to C++ but for general C# applications
        return self._generate_cpp_code(strategy_name, indicators, template, parameters)
    
    def _generate_ninjatrader_code(
        self,
        strategy_name: str,
        indicators: List[Indicator],
        template: Dict,
        parameters: Dict
    ) -> str:
        """Generate NinjaTrader specific code"""
        # Same as C++ for NinjaTrader
        return self._generate_cpp_code(strategy_name, indicators, template, parameters)
    
    def save_strategy(self, strategy: Strategy, output_dir: str = "strategies"):
        """Save generated strategy to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save strategy metadata
        metadata = {
            "name": strategy.name,
            "type": strategy.type.value,
            "description": strategy.description,
            "logic": strategy.logic,
            "parameters": strategy.parameters,
            "indicators": [ind.name for ind in strategy.indicators]
        }
        
        with open(f"{output_dir}/{strategy.name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save code files
        for language, code in strategy.code.items():
            if language == "python":
                ext = "py"
            elif language == "cpp":
                ext = "cpp"
            elif language == "easylanguage":
                ext = "els"
            elif language == "csharp":
                ext = "cs"
            else:
                ext = "txt"
            
            with open(f"{output_dir}/{strategy.name}.{ext}", "w") as f:
                f.write(code)
        
        print(f"Strategy saved to {output_dir}/{strategy.name}_*")
    
    def list_available_indicators(self) -> List[str]:
        """List all available indicators"""
        return list(self.indicators.keys())
    
    def list_available_strategies(self) -> List[str]:
        """List all available strategy templates"""
        return list(self.templates.keys())


# Example usage
if __name__ == "__main__":
    generator = StratenGenerator()
    
    # Generate a mean reversion strategy
    strategy = generator.generate_strategy(
        indicators=["bollinger_bands", "rsi"],
        logic="mean_reversion",
        language="python"
    )
    
    # Save the strategy
    generator.save_strategy(strategy)
    
    print(f"Generated strategy: {strategy.name}")
    print(f"Description: {strategy.description}")
    print(f"Indicators: {[ind.name for ind in strategy.indicators]}")
    print(f"Available code languages: {list(strategy.code.keys())}") 