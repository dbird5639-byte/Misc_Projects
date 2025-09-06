"""
AI-Powered Code Generator for God Mode Development Platform
Generates complete applications from requirements using multiple AI agents.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import ast
import re
from datetime import datetime

from ..ai_agents.agent_orchestrator import AgentOrchestrator, TaskType, TaskPriority


@dataclass
class CodeComponent:
    """Represents a code component."""
    name: str
    type: str  # 'module', 'class', 'function', 'file'
    content: str
    dependencies: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)
    documentation: str = ""


@dataclass
class ProjectStructure:
    """Represents a project structure."""
    name: str
    description: str
    root_dir: str
    components: List[CodeComponent] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    readme: str = ""


class CodeGenerator:
    """
    AI-powered code generator that creates complete applications.
    """
    
    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)
        
        # Templates and patterns
        self.templates = self._load_templates()
        self.code_patterns = self._load_code_patterns()
        
        # Generated code cache
        self.generated_components = {}
        
        self.logger.info("AI Code Generator initialized")
    
    def _load_templates(self) -> Dict[str, Any]:
        """Load code templates."""
        return {
            "fastapi_app": {
                "main.py": """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="{app_name}", description="{description}")

# Models
{models}

# Routes
{routes}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
""",
                "requirements.txt": """
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
""",
                "README.md": """
# {app_name}

{description}

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## API Documentation

Visit http://localhost:8000/docs for interactive API documentation.
"""
            },
            "trading_system": {
                "main.py": """
import asyncio
from trading_engine import TradingEngine
from strategy_manager import StrategyManager
from risk_manager import RiskManager
from data_feed import DataFeed

class TradingSystem:
    def __init__(self):
        self.engine = TradingEngine()
        self.strategy_manager = StrategyManager()
        self.risk_manager = RiskManager()
        self.data_feed = DataFeed()
    
    async def start(self):
        await self.data_feed.connect()
        await self.engine.start()
        await self.strategy_manager.start()
        await self.risk_manager.start()
        
        print("Trading system started successfully")
    
    async def stop(self):
        await self.engine.stop()
        await self.strategy_manager.stop()
        await self.risk_manager.stop()
        await self.data_feed.disconnect()

if __name__ == "__main__":
    system = TradingSystem()
    try:
        asyncio.run(system.start())
    except KeyboardInterrupt:
        asyncio.run(system.stop())
""",
                "trading_engine.py": """
import asyncio
from typing import Dict, Any
from order_manager import OrderManager
from position_manager import PositionManager

class TradingEngine:
    def __init__(self):
        self.order_manager = OrderManager()
        self.position_manager = PositionManager()
        self.running = False
    
    async def start(self):
        self.running = True
        await self.order_manager.start()
        await self.position_manager.start()
    
    async def stop(self):
        self.running = False
        await self.order_manager.stop()
        await self.position_manager.stop()
    
    async def process_signal(self, signal: Dict[str, Any]):
        if not self.running:
            return
        
        # Process trading signal
        order = await self.order_manager.create_order(signal)
        if order:
            await self.position_manager.update_position(order)
""",
                "strategy_manager.py": """
import asyncio
from typing import Dict, List, Any
from base_strategy import BaseStrategy

class StrategyManager:
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.running = False
    
    async def start(self):
        self.running = True
        for strategy in self.strategies.values():
            await strategy.start()
    
    async def stop(self):
        self.running = False
        for strategy in self.strategies.values():
            await strategy.stop()
    
    def add_strategy(self, name: str, strategy: BaseStrategy):
        self.strategies[name] = strategy
    
    async def process_market_data(self, data: Dict[str, Any]):
        if not self.running:
            return
        
        for strategy in self.strategies.values():
            signal = await strategy.analyze(data)
            if signal:
                yield signal
""",
                "risk_manager.py": """
from typing import Dict, Any
from position_manager import PositionManager

class RiskManager:
    def __init__(self):
        self.max_position_size = 0.02  # 2% of portfolio
        self.max_drawdown = 0.20  # 20% max drawdown
        self.position_manager = PositionManager()
    
    async def start(self):
        # Initialize risk management
        pass
    
    async def stop(self):
        # Cleanup
        pass
    
    def check_risk(self, order: Dict[str, Any]) -> bool:
        # Check if order meets risk criteria
        position_size = order.get('size', 0) / self.position_manager.total_portfolio_value
        
        if position_size > self.max_position_size:
            return False
        
        current_drawdown = self.position_manager.calculate_drawdown()
        if current_drawdown > self.max_drawdown:
            return False
        
        return True
    
    def adjust_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        # Adjust order to meet risk requirements
        adjusted_order = order.copy()
        
        max_size = self.position_manager.total_portfolio_value * self.max_position_size
        if adjusted_order.get('size', 0) > max_size:
            adjusted_order['size'] = max_size
        
        return adjusted_order
"""
            }
        }
    
    def _load_code_patterns(self) -> Dict[str, str]:
        """Load common code patterns."""
        return {
            "class_definition": """
class {class_name}:
    def __init__(self{init_params}):
        {init_body}
    
    {methods}
""",
            "function_definition": """
def {function_name}({params}):
    \"\"\"
    {docstring}
    \"\"\"
    {body}
""",
            "async_function": """
async def {function_name}({params}):
    \"\"\"
    {docstring}
    \"\"\"
    {body}
""",
            "api_route": """
@app.{method}("/{endpoint}")
async def {function_name}({params}):
    \"\"\"
    {description}
    \"\"\"
    {body}
    return {response}
""",
            "database_model": """
class {model_name}(Base):
    __tablename__ = "{table_name}"
    
    id = Column(Integer, primary_key=True, index=True)
    {fields}
    
    def __repr__(self):
        return f"<{model_name}(id={{self.id}})>"
"""
        }
    
    async def generate_project(self, requirements: Dict[str, Any]) -> ProjectStructure:
        """Generate a complete project from requirements."""
        self.logger.info(f"Generating project: {requirements.get('name', 'Unknown')}")
        
        # Phase 1: Analyze requirements
        analysis_task_id = self.orchestrator.create_task(
            task_type=TaskType.RESEARCH,
            description="Analyze project requirements and design architecture",
            requirements=requirements,
            priority=TaskPriority.HIGH
        )
        
        analysis_result = await self.orchestrator.execute_task(analysis_task_id)
        
        # Phase 2: Generate project structure
        structure_task_id = self.orchestrator.create_task(
            task_type=TaskType.ARCHITECTURE,
            description="Generate project structure and component design",
            requirements={
                "analysis": analysis_result,
                "requirements": requirements
            },
            priority=TaskPriority.HIGH,
            dependencies=[analysis_task_id]
        )
        
        structure_result = await self.orchestrator.execute_task(structure_task_id)
        
        # Phase 3: Generate code components
        components = await self._generate_components(structure_result, requirements)
        
        # Phase 4: Generate tests
        tests = await self._generate_tests(components, requirements)
        
        # Phase 5: Generate documentation
        documentation = await self._generate_documentation(components, requirements)
        
        # Create project structure
        project = ProjectStructure(
            name=requirements.get("name", "Generated Project"),
            description=requirements.get("description", ""),
            root_dir=f"./generated/{requirements.get('name', 'project').lower().replace(' ', '_')}",
            components=components,
            dependencies=structure_result.get("dependencies", {}),
            configuration=structure_result.get("configuration", {}),
            readme=documentation.get("readme", "")
        )
        
        # Generate files
        await self._generate_files(project)
        
        self.logger.info(f"Project generated successfully: {project.root_dir}")
        return project
    
    async def _generate_components(self, structure: Dict[str, Any], requirements: Dict[str, Any]) -> List[CodeComponent]:
        """Generate code components based on structure."""
        components = []
        
        for component_spec in structure.get("components", []):
            # Generate component code
            code_task_id = self.orchestrator.create_task(
                task_type=TaskType.CODING,
                description=f"Generate code for {component_spec['name']}",
                requirements={
                    "component_spec": component_spec,
                    "requirements": requirements,
                    "patterns": self.code_patterns
                },
                priority=TaskPriority.MEDIUM
            )
            
            code_result = await self.orchestrator.execute_task(code_task_id)
            
            component = CodeComponent(
                name=component_spec["name"],
                type=component_spec["type"],
                content=code_result.get("code", ""),
                dependencies=component_spec.get("dependencies", []),
                imports=code_result.get("imports", []),
                documentation=code_result.get("documentation", "")
            )
            
            components.append(component)
        
        return components
    
    async def _generate_tests(self, components: List[CodeComponent], requirements: Dict[str, Any]) -> List[str]:
        """Generate tests for components."""
        tests = []
        
        for component in components:
            if component.type in ["class", "function", "module"]:
                test_task_id = self.orchestrator.create_task(
                    task_type=TaskType.TESTING,
                    description=f"Generate tests for {component.name}",
                    requirements={
                        "component": component.__dict__,
                        "requirements": requirements
                    },
                    priority=TaskPriority.MEDIUM
                )
                
                test_result = await self.orchestrator.execute_task(test_task_id)
                component.tests = test_result.get("tests", [])
                tests.extend(component.tests)
        
        return tests
    
    async def _generate_documentation(self, components: List[CodeComponent], requirements: Dict[str, Any]) -> Dict[str, str]:
        """Generate documentation for the project."""
        doc_task_id = self.orchestrator.create_task(
            task_type=TaskType.DOCUMENTATION,
            description="Generate comprehensive project documentation",
            requirements={
                "components": [c.__dict__ for c in components],
                "requirements": requirements
            },
            priority=TaskPriority.LOW
        )
        
        doc_result = await self.orchestrator.execute_task(doc_task_id)
        
        return {
            "readme": doc_result.get("readme", ""),
            "api_docs": doc_result.get("api_docs", ""),
            "component_docs": doc_result.get("component_docs", {})
        }
    
    async def _generate_files(self, project: ProjectStructure):
        """Generate actual files on disk."""
        # Create project directory
        project_dir = Path(project.root_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate README
        readme_path = project_dir / "README.md"
        readme_path.write_text(project.readme)
        
        # Generate requirements.txt
        if project.dependencies:
            requirements_path = project_dir / "requirements.txt"
            requirements_content = "\n".join([
                f"{package}>={version}" for package, version in project.dependencies.items()
            ])
            requirements_path.write_text(requirements_content)
        
        # Generate configuration files
        if project.configuration:
            config_path = project_dir / "config.json"
            config_path.write_text(json.dumps(project.configuration, indent=2))
        
        # Generate component files
        for component in project.components:
            file_path = project_dir / f"{component.name.lower().replace(' ', '_')}.py"
            file_path.write_text(component.content)
            
            # Generate test files
            if component.tests:
                test_path = project_dir / f"test_{component.name.lower().replace(' ', '_')}.py"
                test_content = "\n\n".join(component.tests)
                test_path.write_text(test_content)
        
        self.logger.info(f"Files generated in: {project_dir}")
    
    async def generate_from_template(self, template_name: str, parameters: Dict[str, Any]) -> ProjectStructure:
        """Generate a project from a predefined template."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        components = []
        
        # Generate components from template
        for filename, template_content in template.items():
            if filename.endswith('.py'):
                # Generate Python component
                content = template_content.format(**parameters)
                
                component = CodeComponent(
                    name=filename.replace('.py', ''),
                    type="module",
                    content=content,
                    dependencies=[],
                    imports=[],
                    documentation=f"Generated from {template_name} template"
                )
                
                components.append(component)
        
        # Create project structure
        project = ProjectStructure(
            name=parameters.get("app_name", "Template Project"),
            description=parameters.get("description", ""),
            root_dir=f"./generated/{parameters.get('app_name', 'template').lower().replace(' ', '_')}",
            components=components,
            dependencies={},
            configuration={},
            readme=template.get("README.md", "").format(**parameters)
        )
        
        # Generate files
        await self._generate_files(project)
        
        return project
    
    def analyze_code_quality(self, code: str) -> Dict[str, Any]:
        """Analyze the quality of generated code."""
        try:
            # Parse the code
            tree = ast.parse(code)
            
            # Analyze structure
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
            imports_from = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            
            # Check for common issues
            issues = []
            
            # Check for proper docstrings
            for func in functions:
                if not ast.get_docstring(func):
                    issues.append(f"Function '{func.name}' missing docstring")
            
            for cls in classes:
                if not ast.get_docstring(cls):
                    issues.append(f"Class '{cls.name}' missing docstring")
            
            # Check for proper error handling
            try_except_blocks = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
            if len(functions) > 0 and len(try_except_blocks) / len(functions) < 0.3:
                issues.append("Insufficient error handling")
            
            # Calculate metrics
            metrics = {
                "lines_of_code": len(code.split('\n')),
                "classes": len(classes),
                "functions": len(functions),
                "imports": len(imports) + len(imports_from),
                "complexity": self._calculate_complexity(tree),
                "issues": issues,
                "quality_score": max(0, 100 - len(issues) * 10)
            }
            
            return metrics
            
        except SyntaxError as e:
            return {
                "error": f"Syntax error: {e}",
                "quality_score": 0
            }
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    async def optimize_code(self, code: str, requirements: Dict[str, Any]) -> str:
        """Optimize generated code."""
        optimization_task_id = self.orchestrator.create_task(
            task_type=TaskType.OPTIMIZATION,
            description="Optimize code for performance and quality",
            requirements={
                "code": code,
                "requirements": requirements
            },
            priority=TaskPriority.MEDIUM
        )
        
        optimization_result = await self.orchestrator.execute_task(optimization_task_id)
        
        return optimization_result.get("optimized_code", code)
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """Validate generated code."""
        validation_result = {
            "syntax_valid": True,
            "imports_valid": True,
            "style_compliant": True,
            "issues": []
        }
        
        try:
            # Check syntax
            ast.parse(code)
        except SyntaxError as e:
            validation_result["syntax_valid"] = False
            validation_result["issues"].append(f"Syntax error: {e}")
        
        # Check imports
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Basic import validation
                    pass
        except Exception as e:
            validation_result["imports_valid"] = False
            validation_result["issues"].append(f"Import error: {e}")
        
        # Check style (basic checks)
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if len(line) > 120:  # Line length
                validation_result["style_compliant"] = False
                validation_result["issues"].append(f"Line {i}: Line too long")
        
        return validation_result


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize orchestrator (you would need to provide actual config)
        config = {
            "grok3": {"api_key": "test"},
            "gpt45": {"api_key": "test"},
            "claude": {"api_key": "test"}
        }
        
        orchestrator = AgentOrchestrator(config)
        await orchestrator.initialize_agents()
        
        # Initialize code generator
        generator = CodeGenerator(orchestrator)
        
        # Generate a project
        requirements = {
            "name": "AI Trading Bot",
            "description": "A sophisticated algorithmic trading bot",
            "features": ["real-time data", "multiple strategies", "risk management"],
            "technology": "python, fastapi, postgresql"
        }
        
        project = await generator.generate_project(requirements)
        print(f"Project generated: {project.root_dir}")
        
        # Generate from template
        template_params = {
            "app_name": "My FastAPI App",
            "description": "A simple FastAPI application",
            "models": "# Define your Pydantic models here",
            "routes": "# Define your API routes here"
        }
        
        template_project = await generator.generate_from_template("fastapi_app", template_params)
        print(f"Template project generated: {template_project.root_dir}")
    
    asyncio.run(main()) 