import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useSpring, animated, useTransition } from 'react-spring';
import { motion, AnimatePresence } from 'framer-motion';

import './SmartRecommendations.css';

const SmartRecommendations = ({
  userPreferences = {},
  orderHistory = [],
  currentContext = {},
  onRecommendationSelect,
  onPreferenceUpdate
}) => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [preferenceMode, setPreferenceMode] = useState(false);
  const [aiInsights, setAiInsights] = useState({});

  // User interaction tracking
  const [interactionHistory, setInteractionHistory] = useState([]);
  const [preferenceScores, setPreferenceScores] = useState({});

  // Animation states
  const [animatingRecommendations, setAnimatingRecommendations] = useState(false);
  const [highlightedItem, setHighlightedItem] = useState(null);

  // Mock menu data (in real app, this would come from API)
  const menuData = useMemo(() => ({
    appetizers: [
      { id: 'boneless_wings', name: 'Boneless Wings', price: 10.99, tags: ['spicy', 'popular', 'sharing'], 
        nutrition: { calories: 450, protein: 25, carbs: 15 }, allergens: ['soy'], 
        customization: ['Classic Buffalo', 'Honey BBQ', 'Sweet Asian Chile'] },
      { id: 'mozzarella_sticks', name: 'Mozzarella Sticks', price: 8.99, tags: ['cheese', 'vegetarian'], 
        nutrition: { calories: 380, protein: 15, carbs: 25 }, allergens: ['dairy', 'gluten'], 
        customization: ['Marinara', 'Ranch'] },
      { id: 'spinach_artichoke_dip', name: 'Spinach Artichoke Dip', price: 9.99, tags: ['vegetarian', 'sharing'], 
        nutrition: { calories: 320, protein: 12, carbs: 18 }, allergens: ['dairy'], 
        customization: ['Tortilla Chips', 'Pita Bread'] }
    ],
    entrees: [
      { id: 'riblets', name: 'Riblets', price: 15.99, tags: ['bbq', 'meat', 'popular'], 
        nutrition: { calories: 650, protein: 35, carbs: 25 }, allergens: ['gluten'], 
        customization: ['BBQ Sauce', 'Honey Mustard'] },
      { id: 'classic_burger', name: 'Classic Burger', price: 12.99, tags: ['beef', 'classic'], 
        nutrition: { calories: 580, protein: 28, carbs: 35 }, allergens: ['gluten', 'dairy'], 
        customization: ['Well Done', 'Medium Rare', 'Extra Cheese'] },
      { id: 'fettuccine_alfredo', name: 'Fettuccine Alfredo', price: 13.99, tags: ['pasta', 'creamy'], 
        nutrition: { calories: 720, protein: 22, carbs: 45 }, allergens: ['dairy', 'gluten'], 
        customization: ['Chicken', 'Shrimp', 'Extra Sauce'] }
    ],
    desserts: [
      { id: 'chocolate_mousse', name: 'Chocolate Mousse', price: 6.99, tags: ['chocolate', 'sweet'], 
        nutrition: { calories: 280, protein: 8, carbs: 32 }, allergens: ['dairy'], 
        customization: ['Whipped Cream', 'Berries'] },
      { id: 'apple_pie', name: 'Apple Pie', price: 5.99, tags: ['fruit', 'classic'], 
        nutrition: { calories: 320, protein: 4, carbs: 45 }, allergens: ['gluten'], 
        customization: ['Vanilla Ice Cream', 'Caramel'] }
    ]
  }), []);

  // Generate AI recommendations
  useEffect(() => {
    generateRecommendations();
  }, [userPreferences, orderHistory, currentContext]);

  const generateRecommendations = useCallback(async () => {
    setLoading(true);
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const aiRecommendations = await analyzePreferencesAndGenerateRecommendations();
      setRecommendations(aiRecommendations);
      
      // Generate AI insights
      const insights = await generateAIInsights();
      setAiInsights(insights);
      
    } catch (error) {
      console.error('Error generating recommendations:', error);
    } finally {
      setLoading(false);
    }
  }, [userPreferences, orderHistory, currentContext]);

  const analyzePreferencesAndGenerateRecommendations = useCallback(async () => {
    // Analyze user preferences
    const preferenceAnalysis = analyzeUserPreferences();
    
    // Generate recommendations based on analysis
    const allItems = Object.values(menuData).flat();
    const scoredItems = allItems.map(item => ({
      ...item,
      score: calculateRecommendationScore(item, preferenceAnalysis, currentContext)
    }));
    
    // Sort by score and return top recommendations
    return scoredItems
      .sort((a, b) => b.score - a.score)
      .slice(0, 8)
      .map(item => ({
        ...item,
        reasoning: generateReasoning(item, preferenceAnalysis),
        confidence: item.score
      }));
  }, [menuData, userPreferences, orderHistory, currentContext]);

  const analyzeUserPreferences = useCallback(() => {
    const analysis = {
      preferredCategories: {},
      preferredTags: {},
      priceRange: { min: 0, max: 50 },
      dietaryRestrictions: [],
      flavorPreferences: [],
      mealTiming: 'dinner'
    };

    // Analyze order history
    orderHistory.forEach(order => {
      order.items.forEach(item => {
        // Category preference
        const category = getItemCategory(item.id);
        analysis.preferredCategories[category] = (analysis.preferredCategories[category] || 0) + 1;
        
        // Tag preferences
        const itemData = findItemById(item.id);
        if (itemData) {
          itemData.tags.forEach(tag => {
            analysis.preferredTags[tag] = (analysis.preferredTags[tag] || 0) + 1;
          });
        }
      });
    });

    // Analyze user preferences
    if (userPreferences.dietaryRestrictions) {
      analysis.dietaryRestrictions = userPreferences.dietaryRestrictions;
    }

    if (userPreferences.flavorPreferences) {
      analysis.flavorPreferences = userPreferences.flavorPreferences;
    }

    if (userPreferences.priceRange) {
      analysis.priceRange = userPreferences.priceRange;
    }

    return analysis;
  }, [orderHistory, userPreferences]);

  const calculateRecommendationScore = useCallback((item, analysis, context) => {
    let score = 0.5; // Base score

    // Category preference
    const category = getItemCategory(item.id);
    const categoryPreference = analysis.preferredCategories[category] || 0;
    score += categoryPreference * 0.1;

    // Tag preferences
    item.tags.forEach(tag => {
      const tagPreference = analysis.preferredTags[tag] || 0;
      score += tagPreference * 0.05;
    });

    // Price preference
    if (item.price >= analysis.priceRange.min && item.price <= analysis.priceRange.max) {
      score += 0.2;
    }

    // Dietary restrictions
    if (analysis.dietaryRestrictions.length > 0) {
      const hasRestriction = analysis.dietaryRestrictions.some(restriction => 
        item.allergens.includes(restriction)
      );
      if (!hasRestriction) {
        score += 0.3;
      } else {
        score -= 0.5;
      }
    }

    // Context-based scoring
    if (context.mealTime === 'breakfast' && category === 'appetizers') {
      score += 0.1;
    } else if (context.mealTime === 'dinner' && category === 'entrees') {
      score += 0.1;
    }

    // Popularity boost
    if (item.tags.includes('popular')) {
      score += 0.1;
    }

    return Math.min(score, 1.0);
  }, []);

  const generateReasoning = useCallback((item, analysis) => {
    const reasons = [];

    // Category reasoning
    const category = getItemCategory(item.id);
    const categoryPreference = analysis.preferredCategories[category] || 0;
    if (categoryPreference > 0) {
      reasons.push(`You've ordered ${category} items ${categoryPreference} times before`);
    }

    // Tag reasoning
    const preferredTags = item.tags.filter(tag => analysis.preferredTags[tag] > 0);
    if (preferredTags.length > 0) {
      reasons.push(`Matches your preference for ${preferredTags.join(', ')}`);
    }

    // Price reasoning
    if (item.price <= analysis.priceRange.max) {
      reasons.push('Fits your budget');
    }

    // Dietary reasoning
    if (analysis.dietaryRestrictions.length > 0) {
      const safeTags = item.tags.filter(tag => 
        !analysis.dietaryRestrictions.some(restriction => 
          item.allergens.includes(restriction)
        )
      );
      if (safeTags.length > 0) {
        reasons.push('Meets your dietary requirements');
      }
    }

    return reasons.length > 0 ? reasons.join('. ') : 'Based on popular choices';
  }, []);

  const generateAIInsights = useCallback(async () => {
    const insights = {
      orderingPatterns: [],
      dietaryTrends: [],
      priceAnalysis: {},
      recommendations: []
    };

    // Analyze ordering patterns
    if (orderHistory.length > 0) {
      const recentOrders = orderHistory.slice(-5);
      const categories = recentOrders.flatMap(order => 
        order.items.map(item => getItemCategory(item.id))
      );
      
      const categoryCounts = categories.reduce((acc, category) => {
        acc[category] = (acc[category] || 0) + 1;
        return acc;
      }, {});

      insights.orderingPatterns = Object.entries(categoryCounts)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 2)
        .map(([category, count]) => `${category} (${count} times)`);
    }

    // Dietary trends
    if (userPreferences.dietaryRestrictions) {
      insights.dietaryTrends = userPreferences.dietaryRestrictions.map(restriction => 
        `Avoiding ${restriction}`
      );
    }

    // Price analysis
    if (orderHistory.length > 0) {
      const totalSpent = orderHistory.reduce((sum, order) => sum + order.total, 0);
      const avgOrderValue = totalSpent / orderHistory.length;
      insights.priceAnalysis = {
        averageOrder: avgOrderValue,
        totalSpent,
        orderCount: orderHistory.length
      };
    }

    // Personalized recommendations
    insights.recommendations = [
      'Try our new seasonal items',
      'Consider pairing with a beverage',
      'Perfect for sharing with friends'
    ];

    return insights;
  }, [orderHistory, userPreferences]);

  const handleRecommendationClick = useCallback((recommendation) => {
    // Track interaction
    setInteractionHistory(prev => [...prev, {
      type: 'recommendation_click',
      itemId: recommendation.id,
      timestamp: new Date(),
      score: recommendation.score
    }]);

    // Update preference scores
    setPreferenceScores(prev => ({
      ...prev,
      [recommendation.id]: (prev[recommendation.id] || 0) + 1
    }));

    // Call parent handler
    onRecommendationSelect?.(recommendation);

    // Animate selection
    setHighlightedItem(recommendation.id);
    setTimeout(() => setHighlightedItem(null), 1000);
  }, [onRecommendationSelect]);

  const handlePreferenceUpdate = useCallback((preference, value) => {
    onPreferenceUpdate?.({ ...userPreferences, [preference]: value });
  }, [userPreferences, onPreferenceUpdate]);

  const getItemCategory = useCallback((itemId) => {
    for (const [category, items] of Object.entries(menuData)) {
      if (items.find(item => item.id === itemId)) {
        return category;
      }
    }
    return 'other';
  }, [menuData]);

  const findItemById = useCallback((itemId) => {
    for (const items of Object.values(menuData)) {
      const item = items.find(item => item.id === itemId);
      if (item) return item;
    }
    return null;
  }, [menuData]);

  const filteredRecommendations = useMemo(() => {
    if (selectedCategory === 'all') {
      return recommendations;
    }
    return recommendations.filter(item => getItemCategory(item.id) === selectedCategory);
  }, [recommendations, selectedCategory, getItemCategory]);

  // Animation configurations
  const recommendationTransitions = useTransition(filteredRecommendations, {
    from: { opacity: 0, transform: 'translateY(20px)' },
    enter: { opacity: 1, transform: 'translateY(0px)' },
    leave: { opacity: 0, transform: 'translateY(-20px)' },
    config: { tension: 300, friction: 20 }
  });

  const loadingAnimation = useSpring({
    opacity: loading ? 1 : 0,
    transform: loading ? 'scale(1)' : 'scale(0.8)'
  });

  return (
    <div className="smart-recommendations">
      {/* Header */}
      <div className="recommendations-header">
        <h2>🤖 AI Recommendations</h2>
        <p>Personalized suggestions based on your preferences</p>
      </div>

      {/* AI Insights Panel */}
      <div className="ai-insights-panel">
        <h3>💡 AI Insights</h3>
        <div className="insights-grid">
          {aiInsights.orderingPatterns && aiInsights.orderingPatterns.length > 0 && (
            <div className="insight-card">
              <h4>📊 Ordering Patterns</h4>
              <ul>
                {aiInsights.orderingPatterns.map((pattern, index) => (
                  <li key={index}>{pattern}</li>
                ))}
              </ul>
            </div>
          )}

          {aiInsights.dietaryTrends && aiInsights.dietaryTrends.length > 0 && (
            <div className="insight-card">
              <h4>🥗 Dietary Preferences</h4>
              <ul>
                {aiInsights.dietaryTrends.map((trend, index) => (
                  <li key={index}>{trend}</li>
                ))}
              </ul>
            </div>
          )}

          {aiInsights.priceAnalysis && Object.keys(aiInsights.priceAnalysis).length > 0 && (
            <div className="insight-card">
              <h4>💰 Spending Analysis</h4>
              <p>Average Order: ${aiInsights.priceAnalysis.averageOrder?.toFixed(2)}</p>
              <p>Total Spent: ${aiInsights.priceAnalysis.totalSpent?.toFixed(2)}</p>
              <p>Orders: {aiInsights.priceAnalysis.orderCount}</p>
            </div>
          )}

          {aiInsights.recommendations && aiInsights.recommendations.length > 0 && (
            <div className="insight-card">
              <h4>🎯 Suggestions</h4>
              <ul>
                {aiInsights.recommendations.map((rec, index) => (
                  <li key={index}>{rec}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Category Filter */}
      <div className="category-filter">
        <button
          className={`filter-btn ${selectedCategory === 'all' ? 'active' : ''}`}
          onClick={() => setSelectedCategory('all')}
        >
          All
        </button>
        <button
          className={`filter-btn ${selectedCategory === 'appetizers' ? 'active' : ''}`}
          onClick={() => setSelectedCategory('appetizers')}
        >
          Appetizers
        </button>
        <button
          className={`filter-btn ${selectedCategory === 'entrees' ? 'active' : ''}`}
          onClick={() => setSelectedCategory('entrees')}
        >
          Entrees
        </button>
        <button
          className={`filter-btn ${selectedCategory === 'desserts' ? 'active' : ''}`}
          onClick={() => setSelectedCategory('desserts')}
        >
          Desserts
        </button>
      </div>

      {/* Loading State */}
      <animated.div className="loading-container" style={loadingAnimation}>
        {loading && (
          <div className="loading-content">
            <div className="ai-thinking">
              <div className="brain-icon">🧠</div>
              <div className="thinking-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
            <p>AI is analyzing your preferences...</p>
          </div>
        )}
      </animated.div>

      {/* Recommendations Grid */}
      <div className="recommendations-grid">
        <AnimatePresence>
          {recommendationTransitions((style, recommendation) => (
            <motion.div
              key={recommendation.id}
              style={style}
              className={`recommendation-card ${highlightedItem === recommendation.id ? 'highlighted' : ''}`}
              onClick={() => handleRecommendationClick(recommendation)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <div className="card-header">
                <h3>{recommendation.name}</h3>
                <div className="confidence-badge">
                  {Math.round(recommendation.confidence * 100)}% match
                </div>
              </div>

              <div className="card-content">
                <div className="price">${recommendation.price}</div>
                
                <div className="tags">
                  {recommendation.tags.map(tag => (
                    <span key={tag} className="tag">{tag}</span>
                  ))}
                </div>

                <div className="nutrition">
                  <span>🔥 {recommendation.nutrition.calories} cal</span>
                  <span>🥩 {recommendation.nutrition.protein}g protein</span>
                  <span>🍞 {recommendation.nutrition.carbs}g carbs</span>
                </div>

                <div className="reasoning">
                  <p>{recommendation.reasoning}</p>
                </div>

                <div className="customization">
                  <h4>Customization Options:</h4>
                  <div className="options">
                    {recommendation.customization.map(option => (
                      <span key={option} className="option">{option}</span>
                    ))}
                  </div>
                </div>

                {recommendation.allergens.length > 0 && (
                  <div className="allergens">
                    <h4>Allergens:</h4>
                    <div className="allergen-tags">
                      {recommendation.allergens.map(allergen => (
                        <span key={allergen} className="allergen-tag">{allergen}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="card-actions">
                <button className="add-to-order-btn">
                  Add to Order
                </button>
                <button className="learn-more-btn">
                  Learn More
                </button>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Preference Settings */}
      <div className="preference-settings">
        <button
          className="preference-toggle"
          onClick={() => setPreferenceMode(!preferenceMode)}
        >
          {preferenceMode ? 'Hide' : 'Show'} Preference Settings
        </button>

        {preferenceMode && (
          <div className="preferences-panel">
            <h3>Personalize Your Experience</h3>
            
            <div className="preference-group">
              <label>Dietary Restrictions:</label>
              <div className="checkbox-group">
                {['gluten', 'dairy', 'nuts', 'soy'].map(restriction => (
                  <label key={restriction} className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={userPreferences.dietaryRestrictions?.includes(restriction) || false}
                      onChange={(e) => {
                        const current = userPreferences.dietaryRestrictions || [];
                        const updated = e.target.checked
                          ? [...current, restriction]
                          : current.filter(r => r !== restriction);
                        handlePreferenceUpdate('dietaryRestrictions', updated);
                      }}
                    />
                    {restriction}
                  </label>
                ))}
              </div>
            </div>

            <div className="preference-group">
              <label>Price Range:</label>
              <div className="range-slider">
                <input
                  type="range"
                  min="5"
                  max="30"
                  value={userPreferences.priceRange?.max || 20}
                  onChange={(e) => handlePreferenceUpdate('priceRange', { 
                    min: 5, 
                    max: parseInt(e.target.value) 
                  })}
                />
                <span>${userPreferences.priceRange?.max || 20}</span>
              </div>
            </div>

            <div className="preference-group">
              <label>Flavor Preferences:</label>
              <div className="checkbox-group">
                {['spicy', 'sweet', 'savory', 'creamy'].map(flavor => (
                  <label key={flavor} className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={userPreferences.flavorPreferences?.includes(flavor) || false}
                      onChange={(e) => {
                        const current = userPreferences.flavorPreferences || [];
                        const updated = e.target.checked
                          ? [...current, flavor]
                          : current.filter(f => f !== flavor);
                        handlePreferenceUpdate('flavorPreferences', updated);
                      }}
                    />
                    {flavor}
                  </label>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SmartRecommendations; 