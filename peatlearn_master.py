#!/usr/bin/env python3
"""
PeatLearn Master Dashboard - AI-Enhanced Adaptive Learning System
Full integration of all adaptive learning features with live AI profiling
"""

import streamlit as st
import sys
import os
import subprocess
import signal
import time
import requests
from datetime import datetime, timedelta
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from dotenv import load_dotenv
import html
import re

# Load environment variables
load_dotenv()

# --- Orchestrator: run backend servers + Streamlit together when invoked via `python peatlearn_master.py` ---
def _wait_for_health(url: str, timeout_seconds: int = 90) -> bool:
    start = time.time()
    while time.time() - start < timeout_seconds:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def _launch_all():
    print("🚀 Launching PeatLearn: backends + Streamlit...")
    env = os.environ.copy()
    # Mark child Streamlit process to avoid re-launch recursion
    env["RUNNING_UNDER_STREAMLIT"] = "1"

    procs = []
    try:
        # Start API backends
        api_cmd = [sys.executable, "-m", "uvicorn", "inference.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
        adv_cmd = [sys.executable, "-m", "uvicorn", "inference.backend.advanced_app:app", "--host", "0.0.0.0", "--port", "8001"]
        procs.append(subprocess.Popen(api_cmd, env=env))
        procs.append(subprocess.Popen(adv_cmd, env=env))

        # Wait for health
        ok_api = _wait_for_health("http://localhost:8000/api/health", 90)
        print(f"{'✅' if ok_api else '⚠️'} API 8000 health: {'OK' if ok_api else 'not ready'}")
        ok_adv = _wait_for_health("http://localhost:8001/api/health", 90)
        print(f"{'✅' if ok_adv else '⚠️'} Advanced API 8001 health: {'OK' if ok_adv else 'not ready'}")

        # Launch Streamlit for this same script
        st_cmd = ["streamlit", "run", os.path.abspath(__file__)]
        streamlit_proc = subprocess.Popen(st_cmd, env=env)

        # Wait until streamlit exits
        exit_code = streamlit_proc.wait()
        return exit_code
    finally:
        # Cleanup child processes
        for p in procs:
            try:
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        p.kill()
            except Exception:
                pass

# If run directly (not by Streamlit), act as a launcher and exit before any Streamlit UI code runs
if __name__ == "__main__" and os.environ.get("RUNNING_UNDER_STREAMLIT") != "1":
    sys.exit(_launch_all())

# Add src directory to path for our adaptive learning modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our AI-enhanced adaptive learning system
from adaptive_learning.data_logger import DataLogger
from adaptive_learning.ai_profile_analyzer import AIEnhancedProfiler
from adaptive_learning.content_selector import ContentSelector
from adaptive_learning.quiz_generator import QuizGenerator
from adaptive_learning.dashboard import Dashboard
from adaptive_learning.rag_system import RayPeatRAG

# Page configuration
st.set_page_config(
    page_title="PeatLearn - AI-Enhanced Adaptive Learning",
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
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .chat-message {
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: box-shadow 0.3s ease;
    }
    
    .chat-message:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background-color: #667eea;
        color: white;
        margin-left: 2rem;
        border-bottom-right-radius: 0;
    }
    
    .assistant-message {
        background-color: #f8f9fa;
        margin-right: 2rem;
        border-bottom-left-radius: 0;
    }
    
    .rag-answer h3 {
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-size: 1.2em;
    }
    
    .rag-answer p {
        margin-top: 0.3rem;
        margin-bottom: 0.3rem;
        line-height: 1.6;
    }
    
    .rag-answer ul {
        padding-left: 1.5rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .rag-answer li {
        margin-bottom: 0.3rem;
    }
    
    /* Sources styling with hover effect */
    .sources-container {
        margin-top: 1.5rem;
        border-top: 1px solid #eaeaea;
        padding-top: 1rem;
    }
    
    .sources-toggle {
        cursor: pointer;
        color: #667eea;
        font-weight: bold;
        display: inline-block;
        padding: 0.3rem 0.5rem;
        border-radius: 4px;
        transition: background-color 0.3s;
    }
    
    .sources-toggle:hover {
        background-color: #f0f2f5;
    }
    
    .sources-content {
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.3s ease-out;
        margin-top: 0.5rem;
    }
    
    .sources-content ul {
        padding-left: 1.2rem;
        margin: 0.5rem 0;
    }
    
    .sources-content li {
        margin-bottom: 0.4rem;
        font-size: 0.9em;
        color: #555;
    }
    
    /* When sources are expanded */
    .sources-container:hover .sources-content {
        max-height: 500px;
        transition: max-height 0.5s ease-in;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b6b;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .profile-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .mastery-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .struggling { background-color: #ffebee; color: #c62828; }
    .learning { background-color: #fff3e0; color: #ef6c00; }
    .advanced { background-color: #e8f5e8; color: #2e7d32; }
    
    /* Feedback buttons */
    .stButton button {
        padding: 0.2rem 0.5rem;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Helpers ---
def _format_ai_answer(raw: str) -> str:
    """Sanitize and enhance LLM answer for display.

    - Remove any model-emitted HTML tags (e.g., stray </div>)
    - Escape remaining content
    - Convert simple bullets/headings to HTML
    - Render a sources section cleanly if present
    """
    if not raw:
        return ""

    # Strip all tags the model might output
    no_tags = re.sub(r"</?[a-zA-Z][^>]*>", "", raw)

    # Detect a trailing sources section
    sources_html = ""
    body = no_tags
    m = re.search(r"(?:^|\n)\s*(?:Source mapping:|📚\s*Sources[^\n]*:)\s*(.+)$", no_tags, flags=re.IGNORECASE | re.DOTALL)
    if m:
        before = no_tags[:m.start()]
        after = m.group(1)
        items = [html.escape(l.strip(" -\t")) for l in after.splitlines() if l.strip()]
        if items:
            # Create hoverable sources with custom CSS
            sources_list = "".join(f"<li>{it}</li>" for it in items[:10])
            sources_html = """
                <div class="sources-container">
                    <div class="sources-toggle">📚 Sources</div>
                    <div class="sources-content">
                        <ul>{}</ul>
                    </div>
                </div>
            """.format(sources_list)
            body = before.strip()


    esc = html.escape(body)

    # Lightweight markdown-ish formatting
    lines = esc.splitlines()
    out = []
    in_ul = False
    for ln in lines:
        if re.match(r"^\s*[-•]\s+", ln):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{ln.lstrip('-• ').strip()}</li>")
        else:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if re.match(r"^\s*\d+\)\s+", ln) or ln.startswith("## "):
                out.append(f"<h3>{ln.lstrip('0123456789) ').lstrip('# ').strip()}</h3>")
            elif ln.strip():
                out.append(f"<p>{ln}</p>")
    if in_ul:
        out.append("</ul>")

    return "".join(out) + sources_html

# Initialize components
@st.cache_resource
def init_adaptive_system():
    """Initialize the adaptive learning system components"""
    data_logger = DataLogger()
    ai_profiler = AIEnhancedProfiler()
    content_selector = ContentSelector(ai_profiler)
    quiz_generator = QuizGenerator(ai_profiler)
    dashboard = Dashboard()
    rag_system = RayPeatRAG()
    
    return data_logger, ai_profiler, content_selector, quiz_generator, dashboard, rag_system

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []


def get_rag_response(query: str, user_profile: dict = None) -> str:
    """
    Get real RAG response using Gemini 2.5-Flash
    Adapted based on user profile and learning state
    """
    # Initialize RAG system
    rag_system = RayPeatRAG()
    
    # Get AI-powered response
    response = rag_system.get_rag_response(query, user_profile)
    
    return response

def render_user_setup():
    """Render user identification setup"""
    st.markdown("<div class='main-header'><h1>🧠 PeatLearn AI - Adaptive Learning</h1><p>Your Personal Ray Peat Bioenergetics Tutor</p></div>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("👤 User Setup")
        
        # Get user ID
        user_id = st.text_input("Enter your name or ID:", value=st.session_state.get('user_id', ''))
        
        if user_id and user_id != st.session_state.user_id:
            st.session_state.user_id = user_id
            
            # Initialize session for this user
            data_logger, ai_profiler, content_selector, quiz_generator, dashboard, rag_system = init_adaptive_system()
            st.session_state.session_id = data_logger.get_session_id()
            
            # Load existing profile if available
            st.session_state.user_profile = ai_profiler.get_user_profile(user_id)
            
            st.success(f"Welcome, {user_id}! 🎉")
            st.rerun()
        
        if st.session_state.user_id:
            st.write(f"**Current User:** {st.session_state.user_id}")
            if st.session_state.session_id:
                st.write(f"**Session:** {st.session_state.session_id[:8]}...")
            else:
                st.write("**Session:** Not initialized")
            
            # Show AI status
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                st.success("🤖 AI Analysis: Enabled")
            else:
                st.warning("🤖 AI Analysis: Using Fallback Mode")

def render_user_profile():
    """Render user profile and learning analytics"""
    if not st.session_state.user_profile:
        st.info("Start chatting to build your learning profile! 📈")
        return
    
    profile = st.session_state.user_profile
    
    st.subheader("📊 Your Learning Profile")
    
    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Learning State", profile.get('overall_state', 'new').title())
    
    with col2:
        st.metric("Learning Style", profile.get('learning_style', 'balanced').replace('_', ' ').title())
    
    with col3:
        st.metric("Total Interactions", profile.get('total_interactions', 0))
    
    with col4:
        avg_feedback = profile.get('average_feedback', 0)
        st.metric("Avg Feedback", f"{avg_feedback:.1f}" if avg_feedback else "N/A")
    
    # Topic mastery
    topic_mastery = profile.get('topic_mastery', {})
    if topic_mastery:
        st.subheader("🎯 Topic Mastery")
        
        mastery_data = []
        for topic, data in topic_mastery.items():
            mastery_data.append({
                'Topic': topic.title(),
                'State': data.get('state', 'unknown'),
                'Mastery Level': data.get('mastery_level', 0),
                'Interactions': data.get('total_interactions', 0)
            })
        
        df = pd.DataFrame(mastery_data)
        
        # Create mastery chart
        fig = px.bar(df, x='Topic', y='Mastery Level', 
                    color='State', 
                    title="Mastery Levels by Topic",
                    color_discrete_map={
                        'struggling': '#ff6b6b',
                        'learning': '#feca57', 
                        'advanced': '#48db71'
                    })
        st.plotly_chart(fig, use_container_width=True)
        
        # Show mastery badges
        for _, row in df.iterrows():
            badge_class = row['State']
            st.markdown(f"""
                <span class="mastery-badge {badge_class}">
                    {row['Topic']}: {row['State'].title()} ({row['Mastery Level']:.1f})
                </span>
            """, unsafe_allow_html=True)
    
    # AI insights if available
    ai_analysis = profile.get('ai_analysis', {})
    if ai_analysis:
        st.subheader("🤖 AI Insights")
        
        insights = ai_analysis.get('insights', [])
        for insight in insights:
            st.write(f"• {insight}")
        
        learning_velocity = ai_analysis.get('learning_velocity')
        if learning_velocity:
            st.info(f"**Learning Velocity:** {learning_velocity.title()}")

def render_recommendations():
    """Render AI-generated recommendations"""
    if not st.session_state.user_profile:
        return
    
    recommendations = st.session_state.user_profile.get('recommendations', [])
    
    if recommendations:
        st.subheader("💡 Personalized Recommendations")
        
        for rec in recommendations[:3]:  # Show top 3
            priority_color = {
                'high': '#ff6b6b',
                'medium': '#feca57',
                'low': '#74b9ff'
            }.get(rec.get('priority', 'medium'), '#74b9ff')
            
            st.markdown(f"""
                <div class="recommendation-card">
                    <h4 style="color: {priority_color};">
                        {rec.get('priority', 'medium').upper()} PRIORITY: {rec.get('title', 'Recommendation')}
                    </h4>
                    <p>{rec.get('description', '')}</p>
                    {f"<small><i>Reasoning: {rec.get('reasoning', '')}</i></small>" if rec.get('reasoning') else ""}
                    <br><small>Source: {rec.get('source', 'unknown').replace('_', ' ').title()}</small>
                </div>
            """, unsafe_allow_html=True)

def render_chat_interface():
    """Render the main chat interface with AI profiling"""
    data_logger, ai_profiler, content_selector, quiz_generator, dashboard, rag_system = init_adaptive_system()
    
    st.subheader("💬 Chat with Ray Peat AI")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                </div>
            """, unsafe_allow_html=True)
        else:
            cleaned_html = _format_ai_answer(message['content'])
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Ray Peat AI:</strong>
                    <div class="rag-answer">{cleaned_html}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Show feedback buttons for assistant messages
            if 'feedback' not in message:
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button("👍", key=f"up_{len(st.session_state.chat_history)}_{message.get('timestamp', '')}"):
                        handle_feedback(message, 1, data_logger, ai_profiler)
                with col2:
                    if st.button("👎", key=f"down_{len(st.session_state.chat_history)}_{message.get('timestamp', '')}"):
                        handle_feedback(message, -1, data_logger, ai_profiler)
    
    # Chat input
    if prompt := st.chat_input("Ask Ray Peat about bioenergetics, metabolism, hormones..."):
        # Add user message
        user_message = {
            'role': 'user',
            'content': prompt,
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.chat_history.append(user_message)
        
        # Generate AI response using real RAG with Gemini
        with st.spinner("Ray Peat AI is thinking..."):
            response = get_rag_response(prompt, st.session_state.user_profile)
        
        # Add assistant message
        assistant_message = {
            'role': 'assistant', 
            'content': response,
            'timestamp': datetime.now().isoformat(),
            'user_query': prompt
        }
        st.session_state.chat_history.append(assistant_message)
        
        st.rerun()

def handle_feedback(message, feedback_value, data_logger, ai_profiler):
    """Handle user feedback and update profile"""
    if not st.session_state.user_id or not st.session_state.session_id:
        return
    
    # Extract topic from the interaction
    from adaptive_learning.profile_analyzer import TopicExtractor
    topic_extractor = TopicExtractor()
    topic = topic_extractor.get_primary_topic(message.get('user_query', '')) or 'general'
    
    # Log the interaction
    data_logger.log_interaction(
        user_id=st.session_state.user_id,
        session_id=st.session_state.session_id,
        user_query=message.get('user_query', ''),
        llm_response=message.get('content', ''),
        topic=topic,
        user_feedback=feedback_value,
        interaction_type='chat'
    )
    
    # Update user profile with AI analysis
    all_interactions = data_logger._load_interactions()
    user_interactions = all_interactions[all_interactions['user_id'] == st.session_state.user_id].to_dict(orient='records')
    
    # Update profile using AI
    updated_profile = ai_profiler.update_user_profile_with_ai(st.session_state.user_id, user_interactions)
    st.session_state.user_profile = updated_profile
    
    # Mark message as having feedback
    message['feedback'] = feedback_value
    
    # Show success message
    feedback_text = "positive" if feedback_value > 0 else "negative"
    st.success(f"Thanks for the {feedback_text} feedback! Your profile has been updated. 📊")
    
    st.rerun()

def render_quiz_interface():
    """Render personalized quiz interface"""
    if not st.session_state.user_profile:
        st.info("Chat with the AI first to build your profile for personalized quizzes! 💬")
        return
    
    st.subheader("🎯 Personalized Quiz")
    
    data_logger, ai_profiler, content_selector, quiz_generator, dashboard, rag_system = init_adaptive_system()
    
    # Quiz topic selection
    topic_mastery = st.session_state.user_profile.get('topic_mastery', {})
    if topic_mastery:
        topics = list(topic_mastery.keys())
        selected_topic = st.selectbox("Choose a topic for your quiz:", topics)
        
        if st.button("Generate Quiz", type="primary"):
            with st.spinner("Creating your personalized quiz..."):
                quiz = quiz_generator.generate_quiz(
                    st.session_state.user_profile,
                    topic=selected_topic,
                    num_questions=3
                )
            
            if quiz and 'questions' in quiz:
                st.success("Quiz generated! 📝")
                
                # Display quiz questions
                for i, question in enumerate(quiz['questions']):
                    st.write(f"**Question {i+1}:** {question.get('question_text', 'Sample question')}")
                    
                    # Show options if available
                    options = question.get('options', [])
                    if options:
                        for j, option in enumerate(options):
                            st.write(f"  {chr(65+j)}. {option}")
                    
                    st.write(f"**Correct Answer:** {question.get('correct_answer', 'Not specified')}")
                    st.divider()
            else:
                st.error("Failed to generate quiz. Please try again.")

def main():
    """Main application"""
    init_session_state()
    
    # Check if user is set up
    if not st.session_state.user_id:
        render_user_setup()
        st.stop()
    
    # Sidebar user info and navigation
    render_user_setup()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Profile", "🎯 Quiz", "📈 Analytics"])
    
    with tab1:
        render_chat_interface()
        render_recommendations()
    
    with tab2:
        render_user_profile()
    
    with tab3:
        render_quiz_interface()
    
    with tab4:
        st.subheader("📈 Learning Analytics")
        if st.session_state.user_profile:
            # Show detailed analytics
            data_logger, ai_profiler, content_selector, quiz_generator, dashboard, rag_system = init_adaptive_system()
            
            # Load interaction data
            all_interactions = data_logger._load_interactions()
            user_interactions = all_interactions[all_interactions['user_id'] == st.session_state.user_id]
            
            if not user_interactions.empty:
                # Interaction timeline
                user_interactions['timestamp'] = pd.to_datetime(user_interactions['timestamp'])
                daily_interactions = user_interactions.groupby(user_interactions['timestamp'].dt.date).size()
                
                fig = px.line(x=daily_interactions.index, y=daily_interactions.values, 
                            title="Daily Interaction Count")
                fig.update_xaxes(title="Date")
                fig.update_yaxes(title="Interactions")
                st.plotly_chart(fig, use_container_width=True)
                
                # Topic distribution
                topic_counts = user_interactions['topic'].value_counts()
                fig = px.pie(values=topic_counts.values, names=topic_counts.index, 
                           title="Topics Explored")
                st.plotly_chart(fig, use_container_width=True)
                
                # Feedback analysis
                feedback_data = user_interactions['user_feedback'].value_counts()
                if len(feedback_data) > 0:
                    fig = px.bar(x=['Negative', 'Positive'], 
                               y=[feedback_data.get(-1, 0), feedback_data.get(1, 0)],
                               title="Feedback Distribution",
                               color=['Negative', 'Positive'],
                               color_discrete_map={'Negative': '#ff6b6b', 'Positive': '#48db71'})
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No interaction data yet. Start chatting to see analytics!")
        else:
            st.info("No profile data yet. Chat with the AI to build your profile!")

if __name__ == "__main__":
    main()
