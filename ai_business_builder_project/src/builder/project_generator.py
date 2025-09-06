"""
Project Generator Module

Generates project structures and templates for AI-powered businesses.
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ProjectTemplate:
    """Data class for project templates"""
    name: str
    description: str
    category: str
    structure: Dict[str, Any]
    dependencies: List[str]
    setup_instructions: List[str]
    ai_features: List[str]

class ProjectGenerator:
    """
    Generates project structures and templates for AI-powered businesses
    """
    
    def __init__(self):
        """Initialize project generator"""
        self.templates = self._load_project_templates()
        self.base_structures = self._load_base_structures()
        
    def _load_project_templates(self) -> Dict[str, ProjectTemplate]:
        """Load project templates"""
        templates = {
            "ai_web_app": ProjectTemplate(
                name="AI Web Application",
                description="Modern web application with AI features",
                category="Web Applications",
                structure={
                    "frontend": ["src", "public", "components", "pages"],
                    "backend": ["api", "models", "services", "utils"],
                    "ai": ["ai_models", "data_processing", "ml_pipeline"],
                    "config": ["config", "environment"],
                    "docs": ["documentation", "api_docs"]
                },
                dependencies=[
                    "react", "node.js", "python", "tensorflow", "fastapi"
                ],
                setup_instructions=[
                    "Install Node.js and Python",
                    "Set up virtual environment",
                    "Install dependencies",
                    "Configure AI models",
                    "Set up database"
                ],
                ai_features=["Machine Learning", "API Integration", "Data Processing"]
            ),
            "ai_mobile_app": ProjectTemplate(
                name="AI Mobile Application",
                description="Mobile app with AI capabilities",
                category="Mobile Applications",
                structure={
                    "ios": ["ios_app", "models", "views", "controllers"],
                    "android": ["android_app", "activities", "fragments"],
                    "backend": ["api", "ai_services", "database"],
                    "shared": ["shared_models", "utils", "config"]
                },
                dependencies=[
                    "react-native", "python", "tensorflow-lite", "firebase"
                ],
                setup_instructions=[
                    "Install React Native CLI",
                    "Set up development environment",
                    "Configure AI models for mobile",
                    "Set up backend services",
                    "Test on devices"
                ],
                ai_features=["On-device AI", "Cloud AI", "Real-time Processing"]
            ),
            "ai_api_service": ProjectTemplate(
                name="AI API Service",
                description="API service with AI functionality",
                category="AI Services",
                structure={
                    "api": ["endpoints", "middleware", "validation"],
                    "ai": ["models", "inference", "training"],
                    "data": ["processing", "storage", "analytics"],
                    "deployment": ["docker", "kubernetes", "monitoring"]
                },
                dependencies=[
                    "fastapi", "tensorflow", "postgresql", "redis", "docker"
                ],
                setup_instructions=[
                    "Set up Python environment",
                    "Install AI frameworks",
                    "Configure database",
                    "Set up API endpoints",
                    "Deploy with Docker"
                ],
                ai_features=["REST API", "Real-time Inference", "Batch Processing"]
            ),
            "ai_content_generator": ProjectTemplate(
                name="AI Content Generator",
                description="AI-powered content creation platform",
                category="Content Creation",
                structure={
                    "frontend": ["ui", "editor", "preview"],
                    "backend": ["content_engine", "ai_models", "storage"],
                    "ai": ["text_generation", "image_generation", "optimization"],
                    "analytics": ["usage_tracking", "performance", "insights"]
                },
                dependencies=[
                    "react", "python", "openai", "stable-diffusion", "postgresql"
                ],
                setup_instructions=[
                    "Set up development environment",
                    "Configure AI APIs",
                    "Set up content storage",
                    "Implement generation pipeline",
                    "Add analytics tracking"
                ],
                ai_features=["Text Generation", "Image Generation", "Content Optimization"]
            )
        }
        
        return templates
    
    def _load_base_structures(self) -> Dict[str, Dict[str, Any]]:
        """Load base project structures"""
        return {
            "python_backend": {
                "requirements.txt": "# Python dependencies",
                "main.py": "# Main application entry point",
                "config.py": "# Configuration settings",
                "README.md": "# Project documentation",
                "src/": {
                    "__init__.py": "",
                    "api/": {"__init__.py": "", "routes.py": "# API routes"},
                    "models/": {"__init__.py": "", "ai_models.py": "# AI model definitions"},
                    "utils/": {"__init__.py": "", "helpers.py": "# Utility functions"}
                },
                "tests/": {"__init__.py": "", "test_main.py": "# Test cases"},
                "data/": {"raw/": "", "processed/": ""},
                "logs/": ""
            },
            "react_frontend": {
                "package.json": "# Node.js dependencies",
                "public/": {"index.html": "# Main HTML file"},
                "src/": {
                    "App.js": "# Main React component",
                    "index.js": "# React entry point",
                    "components/": {"__init__.py": ""},
                    "pages/": {"__init__.py": ""},
                    "utils/": {"__init__.py": ""}
                },
                "README.md": "# Frontend documentation"
            },
            "ai_integration": {
                "ai_config.py": "# AI model configuration",
                "models/": {
                    "__init__.py": "",
                    "text_model.py": "# Text processing models",
                    "image_model.py": "# Image processing models",
                    "prediction.py": "# Prediction pipeline"
                },
                "data/": {
                    "training/": "",
                    "validation/": "",
                    "test/": ""
                },
                "notebooks/": {"exploration.ipynb": "# Data exploration"}
            }
        }
    
    def generate_project(self, template_name: str, project_name: str, 
                        output_dir: str, custom_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Generate a complete project structure
        
        Args:
            template_name: Name of the template to use
            project_name: Name of the project
            output_dir: Directory to create project in
            custom_config: Custom configuration options
            
        Returns:
            True if successful, False otherwise
        """
        if template_name not in self.templates:
            print(f"Template '{template_name}' not found")
            return False
        
        template = self.templates[template_name]
        project_path = os.path.join(output_dir, project_name)
        
        try:
            # Create project directory
            os.makedirs(project_path, exist_ok=True)
            
            # Generate project structure
            self._create_project_structure(template, project_path, project_name, custom_config)
            
            # Generate configuration files
            self._generate_config_files(template, project_path, project_name, custom_config)
            
            # Generate documentation
            self._generate_documentation(template, project_path, project_name)
            
            print(f"Project '{project_name}' generated successfully at {project_path}")
            return True
            
        except Exception as e:
            print(f"Error generating project: {e}")
            return False
    
    def _create_project_structure(self, template: ProjectTemplate, project_path: str, 
                                 project_name: str, custom_config: Optional[Dict[str, Any]] = None):
        """Create the project directory structure"""
        structure = template.structure
        
        for section, subsections in structure.items():
            section_path = os.path.join(project_path, section)
            os.makedirs(section_path, exist_ok=True)
            
            if isinstance(subsections, list):
                for subsection in subsections:
                    subsection_path = os.path.join(section_path, subsection)
                    os.makedirs(subsection_path, exist_ok=True)
                    
                    # Add __init__.py for Python packages
                    if section in ["backend", "ai", "models", "utils"]:
                        init_file = os.path.join(subsection_path, "__init__.py")
                        if not os.path.exists(init_file):
                            with open(init_file, 'w') as f:
                                f.write(f'"""\n{subsection} module for {project_name}\n"""\n')
    
    def _generate_config_files(self, template: ProjectTemplate, project_path: str, 
                              project_name: str, custom_config: Optional[Dict[str, Any]] = None):
        """Generate configuration files for the project"""
        
        # Generate requirements.txt for Python projects
        if any("python" in dep.lower() for dep in template.dependencies):
            requirements_path = os.path.join(project_path, "requirements.txt")
            with open(requirements_path, 'w') as f:
                f.write("# Python dependencies\n")
                f.write("fastapi==0.104.1\n")
                f.write("uvicorn==0.24.0\n")
                f.write("pydantic==2.5.0\n")
                f.write("python-dotenv==1.0.0\n")
                f.write("requests==2.31.0\n")
                f.write("numpy==1.24.3\n")
                f.write("pandas==2.0.3\n")
                if "tensorflow" in template.dependencies:
                    f.write("tensorflow==2.14.0\n")
                if "openai" in template.dependencies:
                    f.write("openai==1.3.0\n")
        
        # Generate package.json for Node.js projects
        if any("react" in dep.lower() or "node" in dep.lower() for dep in template.dependencies):
            package_json = {
                "name": project_name.lower().replace(" ", "-"),
                "version": "1.0.0",
                "description": template.description,
                "main": "src/index.js",
                "scripts": {
                    "start": "react-scripts start",
                    "build": "react-scripts build",
                    "test": "react-scripts test",
                    "eject": "react-scripts eject"
                },
                "dependencies": {
                    "react": "^18.2.0",
                    "react-dom": "^18.2.0",
                    "react-scripts": "5.0.1"
                },
                "devDependencies": {
                    "@types/react": "^18.2.0",
                    "@types/react-dom": "^18.2.0"
                }
            }
            
            package_path = os.path.join(project_path, "package.json")
            with open(package_path, 'w') as f:
                json.dump(package_json, f, indent=2)
        
        # Generate main configuration file
        config_content = f"""
# Configuration for {project_name}

PROJECT_NAME = "{project_name}"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "{template.description}"

# AI Configuration
AI_MODELS = {{
    "text_model": "gpt-3.5-turbo",
    "image_model": "dall-e-2",
    "embedding_model": "text-embedding-ada-002"
}}

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
DEBUG = True

# Database Configuration
DATABASE_URL = "sqlite:///./{project_name.lower()}.db"

# Security Configuration
SECRET_KEY = "your-secret-key-here"
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

# AI Features
AI_FEATURES = {template.ai_features}
"""
        
        config_path = os.path.join(project_path, "config.py")
        with open(config_path, 'w') as f:
            f.write(config_content)
    
    def _generate_documentation(self, template: ProjectTemplate, project_path: str, project_name: str):
        """Generate project documentation"""
        
        readme_content = f"""# {project_name}

{template.description}

## Features

- **AI Integration**: {', '.join(template.ai_features)}
- **Modern Architecture**: Built with latest technologies
- **Scalable Design**: Ready for production deployment
- **Comprehensive Testing**: Full test coverage

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend)
- Docker (optional)

### Installation

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Node.js dependencies (if applicable):
   ```bash
   npm install
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Run the application:
   ```bash
   python main.py
   ```

## Project Structure

```
{project_name}/
├── api/                 # API endpoints
├── models/              # AI models and data models
├── services/            # Business logic
├── utils/               # Utility functions
├── config.py           # Configuration settings
├── main.py             # Application entry point
└── README.md           # This file
```

## AI Features

{chr(10).join([f"- {feature}" for feature in template.ai_features])}

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Formatting
```bash
black src/
isort src/
```

## Deployment

See deployment documentation in `docs/deployment.md`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details
"""
        
        readme_path = os.path.join(project_path, "README.md")
        with open(readme_path, 'w') as f:
            f.write(readme_content)
    
    def list_templates(self) -> List[Dict[str, str]]:
        """List available project templates"""
        return [
            {
                "name": template.name,
                "description": template.description,
                "category": template.category
            }
            for template in self.templates.values()
        ]
    
    def get_template_details(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a template"""
        if template_name not in self.templates:
            return None
        
        template = self.templates[template_name]
        return {
            "name": template.name,
            "description": template.description,
            "category": template.category,
            "structure": template.structure,
            "dependencies": template.dependencies,
            "setup_instructions": template.setup_instructions,
            "ai_features": template.ai_features
        } 