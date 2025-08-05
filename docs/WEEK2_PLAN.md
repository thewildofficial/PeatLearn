# Week 2: Core AI Components

## 🎯 Objectives
- Implement Personalized Quiz Recommendation System
- Build Learning Style Classifier
- Create content difficulty scoring system

## 📋 Tasks

### 1. Quiz Recommendation System (Day 1-3)
**Goal:** Recommend next quiz/topic based on student performance
- **Input:** Quiz scores, time taken, attempts, topic history
- **Output:** Recommended next topic with difficulty level
- **Algorithm:** Collaborative filtering + rule-based logic

```python
class QuizRecommendationEngine:
    def recommend_next_quiz(self, user_id, performance_history):
        # Analyze weak areas
        weak_topics = self.identify_weak_areas(performance_history)
        
        # Check prerequisites
        missing_prereqs = self.check_prerequisites(weak_topics)
        
        # Recommend based on difficulty progression
        if missing_prereqs:
            return self.recommend_prerequisite(missing_prereqs)
        else:
            return self.recommend_next_level(weak_topics)
```

### 2. Learning Style Classifier (Day 4-5)
**Goal:** Predict learning preference (analytical, practical, conceptual)
- **Input:** Interaction patterns, question types preferred, response times
- **Output:** Learning style classification + content format suggestions

```python
class LearningStyleClassifier:
    def classify_learning_style(self, user_interactions):
        features = self.extract_features(user_interactions)
        # Features: question_complexity_preference, response_time_patterns,
        # content_type_engagement, depth_vs_breadth_preference
        
        style = self.kmeans_model.predict(features)
        return {
            "analytical": 0.7,    # Prefers detailed explanations
            "practical": 0.2,     # Prefers applications
            "conceptual": 0.1     # Prefers big picture
        }
```

### 3. Content Difficulty Scoring (Day 6-7)
**Goal:** Score all Ray Peat content by difficulty level
- **Metrics:** Technical term density, concept complexity, prerequisite depth
- **Output:** Difficulty scores (1-10) for all content pieces

```python
def score_content_difficulty(content):
    technical_terms = count_technical_terms(content)
    concept_density = analyze_concept_density(content) 
    prerequisite_depth = calculate_prerequisites(content)
    
    difficulty = (
        technical_terms * 0.4 +
        concept_density * 0.3 + 
        prerequisite_depth * 0.3
    )
    return min(max(difficulty, 1), 10)
```

## 🛠️ Data Generation
- Simulate 1000+ student interactions
- Generate quiz performance data across Ray Peat topics
- Create realistic learning behavior patterns
