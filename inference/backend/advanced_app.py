#!/usr/bin/env python3
"""
Advanced RAG System with Personalization, RL, and Knowledge Graphs

Integrates all sophisticated ML/AI components:
- Neural Collaborative Filtering for recommendations
- Deep Reinforcement Learning for content sequencing
- Knowledge Graph Neural Networks
- Fine-tuned domain embeddings
- Multi-task learning for quiz generation
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import asyncio

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import numpy as np

from config.settings import settings
from rag.rag_system import RayPeatRAG, RAGResponse

# Import advanced ML components
try:
    from personalization.neural_personalization import (
        AdvancedPersonalizationEngine,
        UserInteraction,
        LearningState,
        personalization_engine
    )
    from personalization.rl_agent import (
        AdaptiveLearningAgent,
        LearningEnvironmentState,
        adaptive_agent
    )
    from personalization.knowledge_graph import (
        AdvancedKnowledgeGraph,
        ray_peat_knowledge_graph
    )
    
    ADVANCED_ML_AVAILABLE = True
    
except ImportError as e:
    print(f"⚠️ Advanced ML components not available: {e}")
    print("📦 Install requirements: pip install -r requirements-advanced.txt")
    ADVANCED_ML_AVAILABLE = False

# Pydantic models for API
class UserProfile(BaseModel):
    user_id: str
    name: str
    email: Optional[str] = None
    learning_style: Optional[str] = "unknown"
    preferences: Dict[str, Any] = {}

class InteractionData(BaseModel):
    user_id: str
    content_id: str
    interaction_type: str
    performance_score: float
    time_spent: float
    difficulty_level: float
    topic_tags: List[str]
    context: Dict[str, Any] = {}

class QuizRequest(BaseModel):
    user_id: str
    topic: Optional[str] = None
    difficulty_preference: Optional[float] = None
    num_questions: int = 5

class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = 10
    exclude_seen: bool = True
    topic_filter: Optional[List[str]] = None

class KnowledgeGraphQuery(BaseModel):
    query: str
    max_expansions: int = 5
    include_related_concepts: bool = True

# Initialize FastAPI app
app = FastAPI(
    title="PeatLearn - Advanced AI Learning Platform",
    description="AI-powered personalized learning system for Ray Peat's bioenergetic knowledge",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
rag_system = RayPeatRAG()

@app.on_event("startup")
async def startup_event():
    """Initialize all advanced ML components on startup."""
    if ADVANCED_ML_AVAILABLE:
        print("🚀 Initializing Advanced ML Components...")
        
        # Initialize personalization engine
        await personalization_engine.initialize_models()
        
        # Initialize RL agent
        print("🤖 RL Agent ready")
        
        # Initialize knowledge graph
        await ray_peat_knowledge_graph.initialize_models()
        
        print("✅ All advanced ML components initialized!")
    else:
        print("⚠️ Running in basic mode without advanced ML features")

# Basic endpoints (existing)
@app.get("/")
async def root():
    return {
        "message": "PeatLearn Advanced AI Learning Platform",
        "version": "2.0.0",
        "features": {
            "basic_rag": True,
            "advanced_ml": ADVANCED_ML_AVAILABLE,
            "personalization": ADVANCED_ML_AVAILABLE,
            "reinforcement_learning": ADVANCED_ML_AVAILABLE,
            "knowledge_graph": ADVANCED_ML_AVAILABLE
        }
    }

@app.get("/api/ask")
async def ask_question(
    q: str = Query(..., description="Question to ask Ray Peat's knowledge"),
    user_id: Optional[str] = Query(None, description="User ID for personalization"),
    max_sources: int = Query(5, description="Maximum number of sources to use"),
    min_similarity: float = Query(0.3, description="Minimum similarity threshold")
):
    """Ask a question with optional personalization."""
    
    try:
        # Basic RAG response
        response = await rag_system.answer_question(q, max_sources, min_similarity)
        
        # Add personalization if available and user provided
        if ADVANCED_ML_AVAILABLE and user_id:
            
            # Expand query using knowledge graph
            expanded_query_data = await ray_peat_knowledge_graph.expand_query_with_concepts(q)
            
            if expanded_query_data['num_expansions'] > 0:
                # Re-run RAG with expanded query
                expanded_response = await rag_system.answer_question(
                    expanded_query_data['expanded_query'], 
                    max_sources, 
                    min_similarity
                )
                
                # Blend responses (simple strategy)
                if expanded_response.confidence > response.confidence:
                    response = expanded_response
                    response.query = f"{q} (expanded with: {', '.join(expanded_query_data['expansion_terms'])})"
            
            # Log interaction for learning
            interaction = UserInteraction(
                user_id=user_id,
                content_id=f"question_{hash(q)}",
                interaction_type="question",
                timestamp=datetime.now(),
                performance_score=response.confidence,
                time_spent=0.0,  # Will be updated by frontend
                difficulty_level=0.5,  # Default
                topic_tags=[],  # Could be extracted from question
                context={"question": q, "sources_used": len(response.sources)}
            )
            
            await personalization_engine.update_user_state(user_id, interaction)
        
        return {
            "question": q,
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": [
                {
                    "source_file": source.source_file,
                    "similarity_score": source.similarity_score,
                    "context": source.context[:200] + "..." if len(source.context) > 200 else source.context
                }
                for source in response.sources
            ],
            "personalized": user_id is not None and ADVANCED_ML_AVAILABLE
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# Advanced ML endpoints
if ADVANCED_ML_AVAILABLE:
    
    @app.post("/api/users/profile")
    async def create_or_update_user_profile(profile: UserProfile):
        """Create or update user profile."""
        # In a real system, this would save to database
        return {"message": f"Profile updated for user {profile.user_id}", "profile": profile}
    
    @app.post("/api/interactions")
    async def log_interaction(interaction: InteractionData):
        """Log a user interaction for learning."""
        
        user_interaction = UserInteraction(
            user_id=interaction.user_id,
            content_id=interaction.content_id,
            interaction_type=interaction.interaction_type,
            timestamp=datetime.now(),
            performance_score=interaction.performance_score,
            time_spent=interaction.time_spent,
            difficulty_level=interaction.difficulty_level,
            topic_tags=interaction.topic_tags,
            context=interaction.context
        )
        
        await personalization_engine.update_user_state(interaction.user_id, user_interaction)
        
        # Update RL agent
        learning_state = LearningEnvironmentState(
            user_id=interaction.user_id,
            current_topic_mastery={},  # Would be populated from user state
            recent_performance=[interaction.performance_score],
            time_in_session=interaction.time_spent,
            difficulty_progression=[interaction.difficulty_level],
            engagement_level=min(1.0, interaction.performance_score + 0.2),
            fatigue_level=max(0.0, interaction.time_spent / 3600 - 0.5),
            topics_covered_today=len(interaction.topic_tags),
            consecutive_correct=0,  # Would track this
            consecutive_incorrect=0,  # Would track this
            preferred_learning_style="unknown",
            context_features=np.zeros(20)  # Placeholder
        )
        
        # Store experience for RL training
        # This is simplified - in practice, you'd track state transitions
        reward = adaptive_agent.calculate_reward(
            learning_state, 0, learning_state, {
                'performance_score': interaction.performance_score,
                'difficulty_rating': interaction.difficulty_level,
                'actual_time': interaction.time_spent,
                'expected_time': 300
            }
        )
        
        await adaptive_agent.update_bandit(0, reward)
        
        return {"message": "Interaction logged successfully", "reward": reward}
    
    @app.post("/api/recommendations")
    async def get_recommendations(request: RecommendationRequest):
        """Get personalized content recommendations."""
        
        recommendations = await personalization_engine.get_content_recommendations(
            request.user_id,
            request.num_recommendations,
            request.exclude_seen
        )
        
        return {
            "user_id": request.user_id,
            "recommendations": [
                {
                    "content_id": content_id,
                    "predicted_score": score,
                    "recommendation_reason": "Neural Collaborative Filtering"
                }
                for content_id, score in recommendations
            ]
        }
    
    @app.post("/api/quiz/generate")
    async def generate_personalized_quiz(request: QuizRequest):
        """Generate a personalized quiz using multi-task neural networks."""
        
        # Get user's learning state
        user_state = personalization_engine.learning_states.get(
            request.user_id,
            LearningState(
                user_id=request.user_id,
                topic_mastery={},
                learning_velocity=0.5,
                preferred_difficulty=0.5,
                learning_style_vector=np.random.normal(0, 0.1, 128),
                attention_span=30.0,
                last_active=datetime.now()
            )
        )
        
        # Generate quiz specification
        content_embedding = np.random.normal(0, 0.1, 768)  # Placeholder
        quiz_spec = await personalization_engine.generate_personalized_quiz(
            request.user_id,
            content_embedding,
            user_state
        )
        
        # Convert to actual quiz questions (simplified)
        quiz_questions = []
        for i in range(request.num_questions):
            question = {
                "question_id": f"q_{i}",
                "question_text": f"Sample question {i+1} about Ray Peat concepts",
                "question_type": quiz_spec['question_type'],
                "difficulty": quiz_spec['difficulty'],
                "predicted_performance": quiz_spec['predicted_performance'],
                "time_limit": quiz_spec['recommended_time_limit'],
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": 0
            }
            quiz_questions.append(question)
        
        return {
            "user_id": request.user_id,
            "quiz_id": f"quiz_{request.user_id}_{datetime.now().isoformat()}",
            "questions": quiz_questions,
            "quiz_metadata": quiz_spec
        }
    
    @app.post("/api/knowledge-graph/query")
    async def query_knowledge_graph(request: KnowledgeGraphQuery):
        """Query the knowledge graph for concept expansion and relationships."""
        
        expanded_query = await ray_peat_knowledge_graph.expand_query_with_concepts(
            request.query,
            request.max_expansions
        )
        
        result = {
            "original_query": request.query,
            "expanded_query": expanded_query['expanded_query'],
            "expansion_terms": expanded_query['expansion_terms'],
            "expansion_sources": expanded_query['expansion_sources'],
            "num_expansions": expanded_query['num_expansions']
        }
        
        if request.include_related_concepts:
            # Find related concepts in graph
            # This would require concept extraction from query first
            pass
        
        return result
    
    @app.get("/api/analytics/user/{user_id}")
    async def get_user_analytics(user_id: str):
        """Get comprehensive analytics for a user."""
        
        user_analytics = personalization_engine.get_user_analytics(user_id)
        
        return {
            "user_analytics": user_analytics,
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/api/analytics/system")
    async def get_system_analytics():
        """Get system-wide analytics."""
        
        agent_analytics = adaptive_agent.get_agent_analytics()
        graph_analytics = ray_peat_knowledge_graph.get_graph_analytics()
        
        return {
            "reinforcement_learning": agent_analytics,
            "knowledge_graph": graph_analytics,
            "total_users": len(personalization_engine.learning_states),
            "system_status": "operational",
            "timestamp": datetime.now().isoformat()
        }
    
    @app.post("/api/rl/train")
    async def train_rl_agent():
        """Manually trigger RL agent training."""
        
        training_results = await adaptive_agent.train_dqn()
        
        return {
            "message": "RL agent training completed",
            "training_loss": training_results['loss'],
            "training_steps": adaptive_agent.steps_done,
            "epsilon": adaptive_agent.epsilon
        }

else:
    # Placeholder endpoints when advanced ML is not available
    @app.post("/api/recommendations")
    async def get_recommendations_basic(request: RecommendationRequest):
        return {
            "error": "Advanced ML features not available",
            "message": "Install requirements-advanced.txt to enable personalization"
        }

# Health check
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "advanced_ml": ADVANCED_ML_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
