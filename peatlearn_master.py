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
from adaptive_learning.topic_model import CorpusTopicModel

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
def _split_answer_and_sources(raw: str) -> tuple[str, list[str]]:
    """Split body and sources list from the model output without escaping.

    Returns (body_markdown, sources_lines)
    """
    if not raw:
        return "", []
    m = re.search(r"(?:^|\n)\s*(?:Source mapping:|📚\s*Sources[^\n]*:)\s*(.+)$", raw, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return raw, []
    body = raw[:m.start()].rstrip()
    tail = m.group(1)
    sources = [l.strip(" -*\t") for l in tail.splitlines() if l.strip()]
    return body, sources

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
    # Load topic model if available
    try:
        topic_model = CorpusTopicModel(model_dir="data/models/topics")
        topic_model.load()
    except Exception:
        topic_model = None
    
    return data_logger, ai_profiler, content_selector, quiz_generator, dashboard, rag_system, topic_model

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
            data_logger, ai_profiler, content_selector, quiz_generator, dashboard, rag_system, topic_model = init_adaptive_system()
            st.session_state.session_id = data_logger.get_session_id()
            
            # Load existing profile if available
            st.session_state.user_profile = ai_profiler.get_user_profile(user_id)
            
            st.success(f"Welcome, {user_id}! 🎉")
            st.rerun()
        
        if st.session_state.user_id:
            st.write(f"**Current User:** {st.session_state.user_id}")
            # Ensure session_id is initialized
            if not st.session_state.get('session_id'):
                try:
                    # Lazily initialize session id if missing
                    DataLogger().get_session_id()
                except Exception:
                    pass
            if st.session_state.session_id:
                st.write(f"**Session:** {st.session_state.session_id[:8]}...")
            else:
                st.write("**Session:** Not initialized")
            
            # Show AI and personalization backend status
            api_key = os.getenv('GEMINI_API_KEY')
            adv_ok = False
            try:
                resp = requests.get("http://localhost:8001/api/health", timeout=2)
                adv_ok = resp.status_code == 200 and resp.json().get('status') == 'healthy'
            except Exception:
                adv_ok = False
            cols = st.columns(2)
            with cols[0]:
                if api_key:
                    st.success("🤖 AI Analysis: Enabled")
                else:
                    st.warning("🤖 AI Analysis: Using Fallback Mode")
            with cols[1]:
                if adv_ok:
                    st.success("🧩 Personalization API: Connected")
                else:
                    st.warning("🧩 Personalization API: Basic Mode")

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
    """Render personalized recommendations via backend."""
    if not st.session_state.get('user_id'):
        return
    st.subheader("💡 Personalized Recommendations")
    try:
        topic_mastery = (st.session_state.user_profile or {}).get('topic_mastery', {})
        topic_filter = list(topic_mastery.keys())[:5] if topic_mastery else None
        payload = {
            "user_id": st.session_state.user_id,
            "num_recommendations": 8,
            "exclude_seen": True,
            "topic_filter": topic_filter,
        }
        r = requests.post("http://localhost:8001/api/recommendations", json=payload, timeout=6)
        if r.status_code == 200:
            data = r.json()
            recs = data.get("recommendations", [])
            if not recs:
                st.info("No recommendations yet. Start interacting to personalize.")
                return
            for rec in recs:
                title = rec.get('title') or rec.get('content_id')
                reason = rec.get('recommendation_reason', '')
                snippet = rec.get('snippet', '')
                st.markdown(f"""
                    <div class="recommendation-card">
                        <h4 style="color:#ff6b6b;">{title}</h4>
                        <p>{snippet}</p>
                        {f"<small><i>{reason}</i></small>" if reason else ""}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Could not fetch recommendations.")
    except Exception as e:
        st.warning(f"Recommendations unavailable: {e}")

def render_chat_interface():
    """Render the main chat interface with AI profiling"""
    data_logger, ai_profiler, content_selector, quiz_generator, dashboard, rag_system, topic_model = init_adaptive_system()
    
    st.subheader("💬 Chat with Ray Peat AI")
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                </div>
            """, unsafe_allow_html=True)
        else:
            body_md, sources = _split_answer_and_sources(message['content'])
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Ray Peat AI:</strong>
                </div>
            """, unsafe_allow_html=True)
            # Render markdown body first (allows headings/lists)
            st.markdown(body_md)
            # Render sources with hover UI if present
            if sources:
                sources_list = "".join(f"<li>{html.escape(it)}</li>" for it in sources[:12])
                st.markdown(f"""
                    <div class="rag-answer">
                        <div class="sources-container">
                            <div class="sources-toggle">📚 Sources</div>
                            <div class="sources-content">
                                <ul>{sources_list}</ul>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Show feedback buttons for assistant messages
            if 'feedback' not in message:
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button("👍", key=f"up_{i}_{message.get('timestamp', '')}"):
                        handle_feedback(message, 1, data_logger, ai_profiler)
                with col2:
                    if st.button("👎", key=f"down_{i}_{message.get('timestamp', '')}"):
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
        
        # Add assistant message (attach parsed sources)
        body_md_tmp, sources_tmp = _split_answer_and_sources(response)
        assistant_message = {
            'role': 'assistant', 
            'content': response,
            'timestamp': datetime.now().isoformat(),
            'user_query': prompt,
            'sources': sources_tmp
        }
        st.session_state.chat_history.append(assistant_message)
        
        st.rerun()

def handle_feedback(message, feedback_value, data_logger, ai_profiler):
    """Handle user feedback and update profile"""
    if not st.session_state.get('user_id'):
        st.warning("Set a user ID first in the sidebar.")
        return
    # Ensure session id exists
    if not st.session_state.get('session_id'):
        try:
            DataLogger().get_session_id()
        except Exception:
            pass
    
    # Extract topic (hybrid): prefer RAG sources vote, fallback to centroid similarity
    assigned_topic = 'general'
    similarity_conf = 0.0
    jargon = 0.0
    try:
        tm = CorpusTopicModel(model_dir="data/models/topics")
        tm.load()
        # Source vote
        srcs = message.get('sources', []) if isinstance(message, dict) else []
        files = []
        import re as _re
        for s in srcs:
            m = _re.search(r"\d+\.\s*([^\(\n]+)", s)
            if m:
                files.append(m.group(1).strip())
        cluster = tm.assign_topic_from_rag_sources(files) if files else None
        # Filter meta-like clusters from source vote
        meta_terms = ["host", "author", "dr ", "dr.", "yeah", "uh", "context", "asks"]
        def is_meta(lbl: str) -> bool:
            l = lbl.lower()
            return any(t in l for t in meta_terms)
        q = message.get('user_query', '')
        if cluster and is_meta(cluster.label):
            cluster = None
        if not cluster:
            cluster = tm.assign_topic_from_text(q)
        if cluster:
            assigned_topic = cluster.label.split(',')[0].strip().lower().replace(' ', '_') or 'general'
            # Compute both metrics regardless of path
            similarity_conf = tm.similarity_to_cluster(q, cluster)
            jargon = tm.jargon_score(q, cluster, top_n=12)
    except Exception:
        from adaptive_learning.profile_analyzer import TopicExtractor
        topic_extractor = TopicExtractor()
        assigned_topic = topic_extractor.get_primary_topic(message.get('user_query', '')) or 'general'
    topic = assigned_topic
    
    # Log the interaction (include sources in context if present)
    sources_list = message.get('sources', []) if isinstance(message, dict) else []
    # Use a fresh logger to avoid any stale state
    _logger = DataLogger()
    _logger.log_interaction(
        user_query=message.get('user_query', ''),
        llm_response=message.get('content', ''),
        topic=topic,
        user_feedback=feedback_value,
        interaction_type='chat',
        context={'sources': sources_list, 'jargon_score': jargon, 'similarity_confidence': similarity_conf}
    )
    try:
        import os as _os
        csv_path = str(_logger.interactions_file)
        size = _os.path.getsize(csv_path)
        st.toast(f"Interaction logged → {csv_path} ({size} bytes)")
    except Exception:
        st.toast("Interaction logged.")

    # Forward interaction to personalization backend to update state
    try:
        perf = 0.9 if feedback_value == 1 else (0.1 if feedback_value == -1 else 0.5)
        first_source = ''
        if sources_list:
            # Try to extract a filename from "1. filename (relevance: x)"
            import re as _re
            m = _re.search(r"\d+\.\s*([^\(\n]+)", sources_list[0])
            if m:
                first_source = m.group(1).strip()
        payload = {
            'user_id': st.session_state.user_id,
            'content_id': first_source or f"chat_{datetime.now().timestamp():.0f}",
            'interaction_type': 'chat',
            'performance_score': perf,
            'time_spent': 0.0,
            'difficulty_level': 0.5,
            'topic_tags': [topic] if topic else [],
            'context': {'sources': sources_list}
        }
        requests.post("http://localhost:8001/api/interactions", json=payload, timeout=3)
    except Exception:
        pass
    
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
    """Render personalized quiz interface (via backend)."""
    if not st.session_state.get('user_id'):
        st.info("Enter your user ID first.")
        return
    st.subheader("🎯 Personalized Quiz")
    topic_mastery = (st.session_state.user_profile or {}).get('topic_mastery', {})
    topics = list(topic_mastery.keys()) if topic_mastery else [
        "thyroid function and metabolism",
        "progesterone and estrogen balance",
        "sugar and cellular energy",
        "carbon dioxide and metabolism",
    ]
    selected_topic = st.selectbox("Choose a topic for your quiz:", topics)
    num_q = st.slider("Number of questions", 3, 10, 5)
    if st.button("Generate Quiz", type="primary"):
        try:
            payload = {"user_id": st.session_state.user_id, "topic": selected_topic, "num_questions": num_q}
            with st.spinner("Creating your personalized quiz..."):
                r = requests.post("http://localhost:8001/api/quiz/generate", json=payload, timeout=20)
            if r.status_code == 200:
                quiz = r.json()
                st.success("Quiz generated! 📝")
                for i, q in enumerate(quiz.get('questions', [])):
                    st.write(f"**Question {i+1}:** {q.get('question_text', '')}")
                    options = q.get('options', [])
                    if options:
                        for j, opt in enumerate(options):
                            st.write(f"  {chr(65+j)}. {opt}")
                    if 'ray_peat_context' in q:
                        with st.expander("Context"):
                            st.write(q['ray_peat_context'])
                    st.divider()
            else:
                st.error(f"Failed to generate quiz: {r.text}")
        except Exception as e:
            st.error(f"Quiz service unavailable: {e}")

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
        if not st.session_state.user_id:
            st.info("Enter your user ID to view analytics.")
        else:
            ok = False
            try:
                r = requests.get(f"http://localhost:8001/api/analytics/user/{st.session_state.user_id}", timeout=8)
                ok = r.status_code == 200
            except Exception:
                ok = False
            if ok:
                data = r.json().get('user_analytics', {})
                if 'error' in data:
                    st.info("No analytics yet. Interact more to build your profile.")
                else:
                    cols = st.columns(3)
                    cols[0].metric("Avg Mastery", f"{data.get('average_mastery',0):.2f}")
                    cols[1].metric("Learning Velocity", f"{data.get('learning_velocity',0):.2f}")
                    cols[2].metric("Preferred Difficulty", f"{data.get('preferred_difficulty',0):.2f}")
                    top_topics = data.get('top_topics', [])
                    if top_topics:
                        if isinstance(top_topics[0], dict):
                            df_top = pd.DataFrame(top_topics)
                            if 'name' in df_top.columns and 'importance' in df_top.columns:
                                df_top.rename(columns={'name':'Topic','importance':'Mastery'}, inplace=True)
                        else:
                            df_top = pd.DataFrame(top_topics, columns=["Topic","Mastery"]) 
                        st.bar_chart(df_top.set_index(df_top.columns[0]))
            else:
                st.info("Analytics service unavailable. Showing local session stats.")
                data_logger, ai_profiler, content_selector, quiz_generator, dashboard, rag_system, topic_model = init_adaptive_system()
                all_interactions = data_logger._load_interactions()
                user_interactions = all_interactions[all_interactions['user_id'] == st.session_state.user_id]
                if not user_interactions.empty:
                    user_interactions['timestamp'] = pd.to_datetime(user_interactions['timestamp'])
                    daily = user_interactions.groupby(user_interactions['timestamp'].dt.date).size()
                    st.line_chart(daily)

if __name__ == "__main__":
    main()
