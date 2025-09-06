"""
AI Assistant Module

This module integrates AI tools to enhance the development process
for algorithmic trading strategies.
"""

import openai
import anthropic
from typing import Dict, List, Any, Optional
import json
import re
from datetime import datetime
import os

class AIAssistant:
    """AI assistant for trading strategy development"""
    
    def __init__(self, openai_api_key: str = "", anthropic_api_key: str = ""):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        
        # Initialize clients
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        if self.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
    
    def analyze_strategy_code(self, code: str) -> Dict[str, Any]:
        """Analyze trading strategy code using AI"""
        prompt = f"""
        Analyze this trading strategy code and provide feedback:
        
        {code}
        
        Please provide:
        1. Code quality assessment
        2. Potential bugs or issues
        3. Performance optimization suggestions
        4. Risk management considerations
        5. Suggested improvements
        """
        
        try:
            if self.openai_api_key:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert algorithmic trading developer."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000
                )
                return {
                    "analysis": response.choices[0].message.content,
                    "model": "GPT-4",
                    "timestamp": datetime.now().isoformat()
                }
            
            elif self.anthropic_api_key:
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return {
                    "analysis": response.content[0].text,
                    "model": "Claude-3",
                    "timestamp": datetime.now().isoformat()
                }
            
            else:
                return {
                    "error": "No AI API keys configured",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "error": f"AI analysis failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_strategy_ideas(self, market_conditions: str, 
                              risk_tolerance: str) -> List[Dict[str, Any]]:
        """Generate trading strategy ideas using AI"""
        prompt = f"""
        Generate 5 algorithmic trading strategy ideas for:
        Market Conditions: {market_conditions}
        Risk Tolerance: {risk_tolerance}
        
        For each strategy, provide:
        1. Strategy name and description
        2. Entry/exit conditions
        3. Risk management approach
        4. Expected performance characteristics
        5. Implementation considerations
        """
        
        try:
            if self.openai_api_key:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert algorithmic trader."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500
                )
                
                # Parse the response to extract strategies
                content = response.choices[0].message.content
                strategies = self._parse_strategy_ideas(content)
                
                return strategies
            
            elif self.anthropic_api_key:
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                content = response.content[0].text
                strategies = self._parse_strategy_ideas(content)
                
                return strategies
            
            else:
                return [{"error": "No AI API keys configured"}]
                
        except Exception as e:
            return [{"error": f"Strategy generation failed: {str(e)}"}]
    
    def _parse_strategy_ideas(self, content: str) -> List[Dict[str, Any]]:
        """Parse AI-generated strategy ideas"""
        strategies = []
        
        # Simple parsing - look for numbered strategies
        lines = content.split('\n')
        current_strategy = {}
        
        for line in lines:
            line = line.strip()
            
            # Look for strategy headers (e.g., "1. Strategy Name")
            if re.match(r'^\d+\.', line):
                if current_strategy:
                    strategies.append(current_strategy)
                current_strategy = {"name": line.split('.', 1)[1].strip()}
            
            # Look for key sections
            elif line.startswith("Description:"):
                current_strategy["description"] = line.split(":", 1)[1].strip()
            elif line.startswith("Entry/Exit:"):
                current_strategy["entry_exit"] = line.split(":", 1)[1].strip()
            elif line.startswith("Risk Management:"):
                current_strategy["risk_management"] = line.split(":", 1)[1].strip()
            elif line.startswith("Expected Performance:"):
                current_strategy["expected_performance"] = line.split(":", 1)[1].strip()
            elif line.startswith("Implementation:"):
                current_strategy["implementation"] = line.split(":", 1)[1].strip()
        
        # Add the last strategy
        if current_strategy:
            strategies.append(current_strategy)
        
        return strategies
    
    def optimize_backtest_parameters(self, strategy_code: str, 
                                   backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize strategy parameters using AI"""
        prompt = f"""
        Analyze these backtest results and suggest parameter optimizations:
        
        Strategy Code:
        {strategy_code}
        
        Backtest Results:
        {json.dumps(backtest_results, indent=2)}
        
        Please suggest:
        1. Parameter adjustments to improve performance
        2. Alternative parameter ranges to test
        3. Risk management improvements
        4. Potential overfitting concerns
        """
        
        try:
            if self.openai_api_key:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert in backtesting and parameter optimization."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000
                )
                
                return {
                    "optimization_suggestions": response.choices[0].message.content,
                    "model": "GPT-4",
                    "timestamp": datetime.now().isoformat()
                }
            
            elif self.anthropic_api_key:
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                return {
                    "optimization_suggestions": response.content[0].text,
                    "model": "Claude-3",
                    "timestamp": datetime.now().isoformat()
                }
            
            else:
                return {"error": "No AI API keys configured"}
                
        except Exception as e:
            return {"error": f"Parameter optimization failed: {str(e)}"}
    
    def generate_documentation(self, code: str, strategy_name: str) -> str:
        """Generate documentation for trading strategy"""
        prompt = f"""
        Generate comprehensive documentation for this trading strategy:
        
        Strategy Name: {strategy_name}
        
        Code:
        {code}
        
        Please include:
        1. Strategy overview and purpose
        2. Input parameters and their effects
        3. Entry and exit conditions
        4. Risk management features
        5. Performance expectations
        6. Usage instructions
        7. Example usage
        """
        
        try:
            if self.openai_api_key:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert technical writer specializing in trading systems."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500
                )
                
                return response.choices[0].message.content
            
            elif self.anthropic_api_key:
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                return response.content[0].text
            
            else:
                return "Documentation generation requires AI API keys"
                
        except Exception as e:
            return f"Documentation generation failed: {str(e)}"
    
    def review_risk_management(self, strategy_code: str, 
                             risk_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Review risk management implementation"""
        prompt = f"""
        Review the risk management implementation for this trading strategy:
        
        Strategy Code:
        {strategy_code}
        
        Risk Settings:
        {json.dumps(risk_settings, indent=2)}
        
        Please assess:
        1. Risk management adequacy
        2. Potential vulnerabilities
        3. Suggested improvements
        4. Position sizing logic
        5. Stop-loss implementation
        6. Portfolio-level risk controls
        """
        
        try:
            if self.openai_api_key:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert in trading risk management."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000
                )
                
                return {
                    "risk_review": response.choices[0].message.content,
                    "model": "GPT-4",
                    "timestamp": datetime.now().isoformat()
                }
            
            elif self.anthropic_api_key:
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                return {
                    "risk_review": response.content[0].text,
                    "model": "Claude-3",
                    "timestamp": datetime.now().isoformat()
                }
            
            else:
                return {"error": "No AI API keys configured"}
                
        except Exception as e:
            return {"error": f"Risk review failed: {str(e)}"}

class CodeGenerator:
    """AI-powered code generator for trading strategies"""
    
    def __init__(self, ai_assistant: AIAssistant):
        self.ai = ai_assistant
    
    def generate_strategy_template(self, strategy_type: str, 
                                 parameters: Dict[str, Any]) -> str:
        """Generate a strategy template using AI"""
        prompt = f"""
        Generate a Python trading strategy template for:
        Strategy Type: {strategy_type}
        Parameters: {json.dumps(parameters, indent=2)}
        
        Include:
        1. Class definition with proper inheritance
        2. Parameter initialization
        3. Signal generation method
        4. Risk management methods
        5. Documentation strings
        6. Example usage
        """
        
        try:
            if self.ai.openai_api_key:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert Python developer specializing in algorithmic trading."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000
                )
                
                return response.choices[0].message.content
            
            elif self.ai.anthropic_api_key:
                response = self.ai.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                return response.content[0].text
            
            else:
                return "# Strategy template generation requires AI API keys"
                
        except Exception as e:
            return f"# Template generation failed: {str(e)}"

def main():
    """Main function for testing AI assistant"""
    # Initialize AI assistant
    ai_assistant = AIAssistant()
    
    # Test strategy code analysis
    sample_code = """
    def calculate_momentum(prices, period=20):
        return (prices[-1] - prices[-period]) / prices[-period]
    
    def generate_signals(prices, threshold=0.02):
        momentum = calculate_momentum(prices)
        if momentum > threshold:
            return 1  # Buy
        elif momentum < -threshold:
            return -1  # Sell
        return 0  # Hold
    """
    
    print("=== AI Strategy Analysis ===")
    analysis = ai_assistant.analyze_strategy_code(sample_code)
    print(json.dumps(analysis, indent=2))
    
    print("\n=== Strategy Ideas ===")
    ideas = ai_assistant.generate_strategy_ideas("Bull market", "Moderate")
    print(json.dumps(ideas, indent=2))
    
    print("\n=== Documentation Generation ===")
    docs = ai_assistant.generate_documentation(sample_code, "Momentum Strategy")
    print(docs)

if __name__ == "__main__":
    main() 