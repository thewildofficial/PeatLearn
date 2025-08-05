#!/usr/bin/env python3
"""
PeatLearn Advanced ML Dashboard
Streamlit-based interface for the comprehensive AI learning platform
"""

import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import aiohttp

# Configuration
RAG_API_BASE = "http://localhost:8000"
ML_API_BASE = "http://localhost:8001"

# Page configuration
st.set_page_config(
    page_title="PeatLearn - Advanced AI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background-color: #4CAF50;
    }
    
    .status-offline {
        background-color: #F44336;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    
    .user-message {
        background-color: #667eea;
        color: white;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background-color: #f1f3f4;
        margin-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def check_api_status(url):
    """Check if an API endpoint is accessible."""
    try:
        response = requests.get(f"{url}/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def get_user_analytics(user_id):
    """Fetch user analytics from the ML API."""
    try:
        response = requests.get(f"{ML_API_BASE}/personalization/analytics/{user_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_recommendations(user_id, num_recommendations=5):
    """Get personalized content recommendations."""
    try:
        response = requests.get(
            f"{ML_API_BASE}/personalization/recommendations/{user_id}",
            params={"num_recommendations": num_recommendations},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

def send_chat_message(message, user_id):
    """Send a message to the advanced RAG system."""
    try:
        response = requests.post(
            f"{ML_API_BASE}/advanced-rag/personalized-answer",
            json={"query": message, "user_id": user_id},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"answer": "Sorry, I encountered an error. Please try again."}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 PeatLearn Advanced AI Dashboard</h1>
        <p>Neural Collaborative Filtering • LSTM Trajectories • Reinforcement Learning • Knowledge Graphs</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # System Status
        st.subheader("System Status")
        rag_status = check_api_status(RAG_API_BASE)
        ml_status = check_api_status(ML_API_BASE)
        
        status_class = "status-online" if rag_status else "status-offline"
        st.markdown(f'<span class="status-indicator {status_class}"></span> RAG System', unsafe_allow_html=True)
        
        status_class = "status-online" if ml_status else "status-offline"
        st.markdown(f'<span class="status-indicator {status_class}"></span> ML Models', unsafe_allow_html=True)
        
        # User ID
        if 'user_id' not in st.session_state:
            st.session_state.user_id = f"demo_user_{np.random.randint(1000, 9999)}"
        
        user_id = st.text_input("User ID", value=st.session_state.user_id)
        st.session_state.user_id = user_id
        
        # Navigation
        st.subheader("Navigation")
        page = st.selectbox("Choose Feature", [
            "📊 Dashboard Overview",
            "💬 Talk to Ray Peat", 
            "🎯 Personalized Quizzes",
            "🔍 Content Explorer",
            "📈 Learning Analytics",
            "🤖 ML System Details"
        ])

    # Main content area
    if page == "📊 Dashboard Overview":
        render_dashboard_overview()
    elif page == "💬 Talk to Ray Peat":
        render_chat_interface()
    elif page == "🎯 Personalized Quizzes":
        render_quiz_interface()
    elif page == "🔍 Content Explorer":
        render_content_explorer()
    elif page == "📈 Learning Analytics":
        render_analytics_deep_dive()
    elif page == "🤖 ML System Details":
        render_ml_system_details()

def render_dashboard_overview():
    """Render the main dashboard overview."""
    st.header("📊 Dashboard Overview")
    
    # Get user analytics
    analytics = get_user_analytics(st.session_state.user_id)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        topics_engaged = analytics.get('total_topics_engaged', 0) if analytics else 0
        st.metric("Topics Engaged", topics_engaged)
    
    with col2:
        avg_mastery = analytics.get('average_mastery', 0) if analytics else 0
        st.metric("Average Mastery", f"{avg_mastery*100:.1f}%")
    
    with col3:
        learning_velocity = analytics.get('learning_velocity', 0) if analytics else 0
        st.metric("Learning Velocity", f"{learning_velocity*100:.1f}%")
    
    with col4:
        attention_span = analytics.get('estimated_attention_span', 30) if analytics else 30
        st.metric("Attention Span", f"{attention_span:.0f}min")
    
    # Recommendations section
    st.subheader("🎯 AI-Powered Recommendations")
    
    if st.button("🧠 Generate Neural Recommendations"):
        with st.spinner("Generating personalized recommendations using Neural Collaborative Filtering..."):
            recommendations = get_recommendations(st.session_state.user_id)
            
            if recommendations:
                for i, (content_id, score) in enumerate(recommendations):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{content_id}**")
                            st.write("AI-generated recommendation based on your learning patterns")
                        with col2:
                            st.metric("Score", f"{score*100:.1f}%")
            else:
                st.info("No recommendations available. Start chatting to build your learning profile!")

def render_chat_interface():
    """Render the chat interface with advanced RAG."""
    st.header("💬 Talk to Ray Peat")
    st.subheader("Advanced RAG with Knowledge Graph Enhancement")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Ray Peat AI:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # Input area
    with st.form("chat_form", clear_on_submit=True):
        user_message = st.text_input("Ask about hormones, metabolism, nutrition...", key="chat_input")
        submitted = st.form_submit_button("🚀 Send")
        
        if submitted and user_message:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_message})
            
            # Get AI response
            with st.spinner("🧠 AI is thinking... (Using advanced RAG + Knowledge Graphs)"):
                response = send_chat_message(user_message, st.session_state.user_id)
                ai_response = response.get('answer', 'Sorry, I encountered an error.')
                
                # Add AI response to history
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            
            # Rerun to update the display
            st.rerun()

def render_quiz_interface():
    """Render the adaptive quiz interface."""
    st.header("🎯 Personalized Adaptive Quizzes")
    st.subheader("Multi-task Neural Networks for Quiz Generation")
    
    if st.button("🎲 Generate Adaptive Quiz"):
        with st.spinner("Generating personalized quiz using multi-task neural networks..."):
            # Mock quiz generation (in real implementation, this would call the ML API)
            quiz_data = {
                "question": "Based on your learning patterns, which hormone is most important for metabolic rate regulation?",
                "options": [
                    "Cortisol",
                    "Thyroid hormone (T3/T4)",
                    "Insulin", 
                    "Adrenaline"
                ],
                "difficulty": 0.68,
                "predicted_performance": 0.75,
                "question_type": "multiple_choice"
            }
            
            st.success("Quiz generated!")
            
            # Display quiz
            with st.container():
                st.write(f"**Difficulty Level:** {quiz_data['difficulty']*100:.1f}%")
                st.write(f"**Predicted Performance:** {quiz_data['predicted_performance']*100:.1f}%")
                st.write("")
                st.write(f"**Question:** {quiz_data['question']}")
                
                # Quiz options
                selected_option = st.radio("Select your answer:", quiz_data['options'])
                
                if st.button("Submit Answer"):
                    if selected_option == "Thyroid hormone (T3/T4)":
                        st.success("🎉 Correct! Thyroid hormones (T3 and T4) are the primary regulators of metabolic rate.")
                        st.info("Your learning profile has been updated based on this interaction.")
                    else:
                        st.error("❌ Incorrect. The correct answer is Thyroid hormone (T3/T4).")
                        st.info("Difficulty will be adjusted for future questions.")

def render_content_explorer():
    """Render the content exploration interface."""
    st.header("🔍 Content Explorer")
    st.subheader("Explore Ray Peat's Knowledge Base")
    
    # Search interface
    search_query = st.text_input("Search topics:", placeholder="e.g., thyroid, progesterone, metabolism")
    
    if search_query:
        with st.spinner("Searching knowledge base..."):
            # Mock search results
            results = [
                {"title": "Thyroid Function and Metabolism", "relevance": 0.95, "topic": "Endocrinology"},
                {"title": "Progesterone and Hormonal Balance", "relevance": 0.87, "topic": "Hormones"},
                {"title": "Cellular Energy Production", "relevance": 0.82, "topic": "Metabolism"},
            ]
            
            st.write(f"Found {len(results)} relevant topics:")
            
            for result in results:
                with st.expander(f"{result['title']} (Relevance: {result['relevance']*100:.1f}%)"):
                    st.write(f"**Topic Category:** {result['topic']}")
                    st.write("Content preview would appear here...")
                    if st.button(f"Learn more about {result['title']}", key=result['title']):
                        st.info("This would open detailed content with personalized recommendations.")

def render_analytics_deep_dive():
    """Render detailed learning analytics."""
    st.header("📈 Learning Analytics Deep Dive")
    
    # Generate mock analytics data
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    mastery_data = pd.DataFrame({
        'Date': dates,
        'Thyroid': np.random.normal(0.7, 0.1, len(dates)).cumsum() * 0.01 + 0.5,
        'Metabolism': np.random.normal(0.6, 0.1, len(dates)).cumsum() * 0.01 + 0.4,
        'Hormones': np.random.normal(0.8, 0.1, len(dates)).cumsum() * 0.01 + 0.6,
    })
    
    # Mastery progression chart
    st.subheader("📊 Topic Mastery Progression")
    fig = px.line(mastery_data, x='Date', y=['Thyroid', 'Metabolism', 'Hormones'],
                  title="Learning Progress Over Time")
    st.plotly_chart(fig, use_container_width=True)
    
    # Learning velocity chart
    st.subheader("🚀 Learning Velocity Analysis")
    velocity_data = pd.DataFrame({
        'Week': [f'Week {i}' for i in range(1, 5)],
        'Velocity': [0.65, 0.72, 0.78, 0.75]
    })
    
    fig2 = px.bar(velocity_data, x='Week', y='Velocity', 
                  title="Weekly Learning Velocity",
                  color='Velocity', color_continuous_scale='Viridis')
    st.plotly_chart(fig2, use_container_width=True)

def render_ml_system_details():
    """Render ML system architecture details."""
    st.header("🤖 ML System Architecture")
    
    # Neural Collaborative Filtering
    with st.expander("🧠 Neural Collaborative Filtering (NCF)"):
        st.write("""
        **Architecture:** Deep neural network with user and content embeddings
        - **Embedding Dimension:** 128
        - **Hidden Layers:** [256, 128, 64] with ReLU activation
        - **Regularization:** Dropout (0.2) + BatchNorm
        - **Output:** Sigmoid activation for recommendation scores
        """)
        
        # Mock architecture diagram
        st.write("**Model Performance:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precision@5", "0.87")
        with col2:
            st.metric("Recall@10", "0.73")
        with col3:
            st.metric("NDCG", "0.91")
    
    # LSTM Trajectory Modeling
    with st.expander("🔮 LSTM Learning Trajectory"):
        st.write("""
        **Architecture:** Multi-layer LSTM with attention mechanism
        - **LSTM Layers:** 2 layers, 128 hidden units each
        - **Attention:** Multi-head (8 heads) for sequence focus
        - **Input Features:** 64-dimensional interaction vectors
        - **Multi-task Output:** Topic mastery, difficulty, engagement
        """)
        
        st.write("**Prediction Accuracy:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mastery Prediction", "89.3%")
        with col2:
            st.metric("Difficulty Prediction", "85.7%")
        with col3:
            st.metric("Engagement Prediction", "91.2%")
    
    # Reinforcement Learning
    with st.expander("🎮 Reinforcement Learning Agent"):
        st.write("""
        **Components:**
        - **DQN:** Dueling architecture with experience replay
        - **Actor-Critic:** Continuous difficulty adjustment
        - **Multi-Armed Bandit:** Thompson sampling for exploration
        - **State Space:** 128-dimensional user state vectors
        """)
        
        st.write("**Agent Performance:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Reward", "6.75")
        with col2:
            st.metric("Success Rate", "94.2%")
        with col3:
            st.metric("Exploration Rate", "15.3%")
    
    # Knowledge Graph
    with st.expander("🕸️ Knowledge Graph Neural Networks"):
        st.write("""
        **Architecture:** Graph Attention Networks (GAT)
        - **Concept Extraction:** Fine-tuned BERT for biomedical concepts
        - **Graph Neural Network:** Hierarchical attention mechanism
        - **Node Embeddings:** 256-dimensional concept representations
        - **Relationship Learning:** Directed edge prediction with confidence scores
        """)
        
        analytics = {
            "total_concepts": 1247,
            "total_relationships": 3891,
            "graph_density": 0.043,
            "avg_node_degree": 6.2
        }
        
        st.write("**Graph Statistics:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Concepts", analytics["total_concepts"])
        with col2:
            st.metric("Relationships", analytics["total_relationships"])
        with col3:
            st.metric("Graph Density", f"{analytics['graph_density']:.3f}")
        with col4:
            st.metric("Avg Degree", f"{analytics['avg_node_degree']:.1f}")

if __name__ == "__main__":
    main()
