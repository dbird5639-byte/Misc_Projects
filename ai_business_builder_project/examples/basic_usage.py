"""
Basic Usage Example for AI Business Builder Project

This example demonstrates how to use the main components of the project
to analyze markets, generate ideas, and build AI-powered businesses.
"""

import sys
import os
import json
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# from analyzer.app_store_analyzer import AppStoreAnalyzer
# from analyzer.market_research import MarketResearch
# from analyzer.idea_generator import IdeaGenerator
# from builder.project_generator import ProjectGenerator
# from utils.ai_tools import AITools

def main():
    """Main example function"""
    print("🚀 AI Business Builder - Basic Usage Example")
    print("=" * 50)
    
    # Initialize components
    print("\n1. Initializing components...")
    app_analyzer = AppStoreAnalyzer()
    market_research = MarketResearch()
    idea_generator = IdeaGenerator()
    project_generator = ProjectGenerator()
    ai_tools = AITools()
    
    # Example 1: Analyze App Store Categories
    print("\n2. Analyzing App Store Categories...")
    categories = app_analyzer.get_categories()
    print(f"Available categories: {', '.join(categories[:5])}...")
    
    # Analyze a specific category
    category = "Education"
    analysis = app_analyzer.analyze_category(category)
    print(f"\nAnalysis for '{category}':")
    print(f"  Market Size: {analysis.get('market_size', 'Unknown')}")
    print(f"  Competition Level: {analysis.get('competition_level', 'Unknown')}")
    print(f"  Revenue Potential: {analysis.get('revenue_potential', 'Unknown')}")
    print(f"  AI Opportunities: {', '.join(analysis.get('ai_opportunities', [])[:3])}")
    
    # Example 2: Find Best Opportunities
    print("\n3. Finding Best Business Opportunities...")
    opportunities = app_analyzer.find_best_opportunities(max_categories=3)
    
    print("Top 3 opportunities:")
    for i, opp in enumerate(opportunities, 1):
        print(f"  {i}. {opp['category']} (Score: {opp['score']})")
        print(f"     Market Size: {opp['market_size']}")
        print(f"     Competition: {opp['competition_level']}")
        print(f"     Revenue Potential: {opp['revenue_potential']}")
    
    # Example 3: Market Research
    print("\n4. Conducting Market Research...")
    market_analysis = market_research.analyze_market_size(category, "Students and educators")
    competition_analysis = market_research.analyze_competition(category)
    
    print(f"Market Analysis for '{category}':")
    print(f"  Market Size: {market_analysis['market_size']} ({market_analysis['market_value']})")
    print(f"  Growth Rate: {market_analysis['growth_rate']}")
    print(f"  Competition Level: {competition_analysis['competition_level']}")
    print(f"  Entry Difficulty: {competition_analysis['entry_difficulty']}")
    
    # Example 4: Validate Business Idea
    print("\n5. Validating Business Idea...")
    idea = "AI-powered language learning app with personalized tutoring"
    validation = market_research.validate_business_idea(idea, category, "Language learners")
    
    print(f"Business Idea: {idea}")
    print(f"Validation Score: {validation['validation_score']}/5.0")
    print(f"Overall Verdict: {validation['overall_verdict']}")
    print("Recommendations:")
    for rec in validation['recommendations'][:3]:
        print(f"  - {rec}")
    
    # Example 5: Generate Business Ideas
    print("\n6. Generating Business Ideas...")
    ideas = idea_generator.generate_ideas(category, count=3)
    
    print(f"Generated ideas for '{category}':")
    for i, idea in enumerate(ideas, 1):
        print(f"\n  {i}. {idea.title}")
        print(f"     Description: {idea.description}")
        print(f"     Target Audience: {idea.target_audience}")
        print(f"     Revenue Model: {idea.revenue_model}")
        print(f"     AI Features: {', '.join(idea.ai_features)}")
    
    # Example 6: AI Tools Integration
    print("\n7. Using AI Tools...")
    
    # Generate text
    text_response = ai_tools.generate_text(
        "Create a business plan for an AI-powered education app",
        service="openai"
    )
    print(f"AI Text Generation: {text_response['text'][:100]}...")
    
    # Analyze sentiment
    sentiment = ai_tools.analyze_sentiment(
        "This AI education app is amazing and very helpful for learning!"
    )
    print(f"Sentiment Analysis: {sentiment['sentiment']} (confidence: {sentiment['confidence']})")
    
    # Generate business ideas with AI
    ai_ideas = ai_tools.generate_business_ideas("Education", "Language Learning")
    print(f"AI-Generated Ideas: {len(ai_ideas)} ideas created")
    
    # Example 7: Project Generation
    print("\n8. Generating Project Structure...")
    templates = project_generator.list_templates()
    print("Available project templates:")
    for template in templates:
        print(f"  - {template['name']}: {template['description']}")
    
    # Example 8: Cost Estimation
    print("\n9. Estimating Development Costs...")
    features = ["AI Integration", "Mobile App", "Real-time Features"]
    cost_estimate = ai_tools.estimate_development_cost(features, "medium")
    
    print(f"Features: {', '.join(features)}")
    print(f"Estimated Cost: ${cost_estimate['estimated_cost']:,.2f}")
    print(f"Timeline: {cost_estimate['timeline_estimate']}")
    print("Cost Breakdown:")
    for category, amount in cost_estimate['cost_breakdown'].items():
        print(f"  {category.title()}: ${amount:,.2f}")
    
    # Example 9: Export Results
    print("\n10. Exporting Results...")
    
    # Create results directory
    results_dir = "example_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Export analysis
    app_analyzer.export_analysis(
        category, 
        os.path.join(results_dir, f"{category.lower()}_analysis.json")
    )
    
    # Export ideas
    idea_generator.export_ideas(
        ideas, 
        os.path.join(results_dir, f"{category.lower()}_ideas.json")
    )
    
    print(f"Results exported to '{results_dir}' directory")
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ Example completed successfully!")
    print("\nKey Takeaways:")
    print("  • Use AppStoreAnalyzer to identify market opportunities")
    print("  • Use MarketResearch to validate business ideas")
    print("  • Use IdeaGenerator to create innovative concepts")
    print("  • Use AITools to enhance your business with AI")
    print("  • Use ProjectGenerator to build your project structure")
    print("\nNext Steps:")
    print("  1. Choose a category that interests you")
    print("  2. Analyze the market and competition")
    print("  3. Generate and validate business ideas")
    print("  4. Use AI tools to enhance your concept")
    print("  5. Generate project structure and start building!")
    
    print(f"\nExample completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 