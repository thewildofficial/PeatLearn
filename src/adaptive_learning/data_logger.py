#!/usr/bin/env python3
"""
Data Logger for PeatLearn Adaptive Learning System
Tracks user interactions, manages sessions, and stores feedback data
"""

import csv
import pandas as pd
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import streamlit as st

class DataLogger:
    """
    Handles logging of user interactions, feedback, and learning data
    """
    
    def __init__(self, data_dir: str = "data/user_interactions"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.interactions_file = self.data_dir / "interactions.csv"
        self.profiles_file = self.data_dir / "user_profiles.json"
        self.quiz_results_file = self.data_dir / "quiz_results.csv"
        
        # Initialize files if they don't exist
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize CSV and JSON files with proper headers"""
        
        # Initialize interactions CSV
        if not self.interactions_file.exists():
            with open(self.interactions_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'user_id', 'session_id', 'timestamp', 'user_query', 
                    'llm_response', 'topic', 'user_feedback', 'interaction_type',
                    'response_time', 'context'
                ])
        
        # Initialize quiz results CSV
        if not self.quiz_results_file.exists():
            with open(self.quiz_results_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'user_id', 'session_id', 'timestamp', 'quiz_id', 'topic',
                    'questions_total', 'questions_correct', 'score_percentage',
                    'difficulty_level', 'time_taken_seconds'
                ])
        
        # Initialize user profiles JSON
        if not self.profiles_file.exists():
            initial_data = {
                "profiles": {},
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
            with open(self.profiles_file, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2)
    
    def get_session_id(self) -> str:
        """Get or create session ID for current Streamlit session"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id
    
    def get_user_id(self) -> str:
        """Get user ID from session state"""
        return st.session_state.get('user_id', 'anonymous')
    
    def log_interaction(self, 
                       user_query: str,
                       llm_response: str,
                       topic: str = None,
                       user_feedback: int = None,
                       interaction_type: str = "chat",
                       response_time: float = None,
                       context: Dict[str, Any] = None) -> None:
        """
        Log a user interaction to the CSV file
        
        Args:
            user_query: The question/query from the user
            llm_response: The AI's response
            topic: Extracted topic (optional)
            user_feedback: Rating (1 for thumbs up, -1 for thumbs down, None for no feedback)
            interaction_type: Type of interaction (chat, quiz, search, etc.)
            response_time: Time taken to generate response in seconds
            context: Additional context data as dictionary
        """
        
        timestamp = datetime.now().isoformat()
        user_id = self.get_user_id()
        session_id = self.get_session_id()
        
        # Convert context dict to JSON string
        context_str = json.dumps(context) if context else ""
        
        # Append to CSV file
        with open(self.interactions_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                user_id, session_id, timestamp, user_query,
                llm_response, topic, user_feedback, interaction_type,
                response_time, context_str
            ])
    
    def log_feedback(self, 
                     interaction_index: int,
                     feedback: int) -> None:
        """
        Update feedback for a specific interaction
        
        Args:
            interaction_index: Index of the interaction in session history
            feedback: 1 for thumbs up, -1 for thumbs down
        """
        
        # For now, store feedback in session state
        # In production, you'd want to update the CSV file
        if 'interaction_feedback' not in st.session_state:
            st.session_state.interaction_feedback = {}
        
        st.session_state.interaction_feedback[interaction_index] = feedback
        
        # Also log as a new interaction
        self.log_interaction(
            user_query=f"Feedback for interaction {interaction_index}",
            llm_response="",
            user_feedback=feedback,
            interaction_type="feedback"
        )
    
    def log_quiz_result(self,
                       quiz_id: str,
                       topic: str,
                       questions_total: int,
                       questions_correct: int,
                       difficulty_level: float,
                       time_taken_seconds: float) -> None:
        """
        Log quiz results to CSV file
        
        Args:
            quiz_id: Unique identifier for the quiz
            topic: Topic of the quiz
            questions_total: Total number of questions
            questions_correct: Number of correct answers
            difficulty_level: Difficulty level (0.0 to 1.0)
            time_taken_seconds: Time taken to complete quiz
        """
        
        timestamp = datetime.now().isoformat()
        user_id = self.get_user_id()
        session_id = self.get_session_id()
        score_percentage = (questions_correct / questions_total * 100) if questions_total > 0 else 0
        
        with open(self.quiz_results_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                user_id, session_id, timestamp, quiz_id, topic,
                questions_total, questions_correct, score_percentage,
                difficulty_level, time_taken_seconds
            ])
    
    def get_user_interactions(self, user_id: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve user interactions from CSV file
        
        Args:
            user_id: Filter by specific user (default: current user)
            limit: Maximum number of interactions to return
            
        Returns:
            List of interaction dictionaries
        """
        
        if user_id is None:
            user_id = self.get_user_id()
        
        interactions = []
        
        try:
            with open(self.interactions_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['user_id'] == user_id:
                        # Parse context JSON
                        try:
                            row['context'] = json.loads(row['context']) if row['context'] else {}
                        except json.JSONDecodeError:
                            row['context'] = {}
                        
                        # Convert feedback to int if present
                        if row['user_feedback']:
                            try:
                                row['user_feedback'] = int(row['user_feedback'])
                            except ValueError:
                                row['user_feedback'] = None
                        else:
                            row['user_feedback'] = None
                        
                        interactions.append(row)
        
        except FileNotFoundError:
            return []
        
        # Apply limit if specified
        if limit:
            interactions = interactions[-limit:]
        
        return interactions

    def _load_interactions(self) -> pd.DataFrame:
        """Load all interactions into a pandas DataFrame for analytics convenience."""
        try:
            df = pd.read_csv(self.interactions_file)
            # Normalize types
            if 'timestamp' in df.columns:
                # leave as string; caller can parse to datetime
                pass
            return df
        except Exception:
            return pd.DataFrame(columns=[
                'user_id','session_id','timestamp','user_query','llm_response','topic','user_feedback','interaction_type','response_time','context'
            ])
    
    def get_quiz_results(self, user_id: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve quiz results for a user
        
        Args:
            user_id: Filter by specific user (default: current user)
            
        Returns:
            List of quiz result dictionaries
        """
        
        if user_id is None:
            user_id = self.get_user_id()
        
        results = []
        
        try:
            with open(self.quiz_results_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['user_id'] == user_id:
                        # Convert numeric fields
                        for field in ['questions_total', 'questions_correct']:
                            try:
                                row[field] = int(row[field])
                            except ValueError:
                                row[field] = 0
                        
                        for field in ['score_percentage', 'difficulty_level', 'time_taken_seconds']:
                            try:
                                row[field] = float(row[field])
                            except ValueError:
                                row[field] = 0.0
                        
                        results.append(row)
        
        except FileNotFoundError:
            return []
        
        return results
    
    def get_interaction_stats(self, user_id: str = None) -> Dict[str, Any]:
        """
        Get basic statistics about user interactions
        
        Args:
            user_id: Filter by specific user (default: current user)
            
        Returns:
            Dictionary with interaction statistics
        """
        
        interactions = self.get_user_interactions(user_id)
        quiz_results = self.get_quiz_results(user_id)
        
        # Calculate basic stats
        total_interactions = len(interactions)
        positive_feedback = sum(1 for i in interactions if i['user_feedback'] == 1)
        negative_feedback = sum(1 for i in interactions if i['user_feedback'] == -1)
        
        # Topic distribution
        topics = [i['topic'] for i in interactions if i['topic']]
        topic_counts = {}
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Quiz stats
        total_quizzes = len(quiz_results)
        avg_quiz_score = sum(r['score_percentage'] for r in quiz_results) / total_quizzes if total_quizzes > 0 else 0
        
        return {
            'total_interactions': total_interactions,
            'positive_feedback': positive_feedback,
            'negative_feedback': negative_feedback,
            'feedback_ratio': positive_feedback / (positive_feedback + negative_feedback) if (positive_feedback + negative_feedback) > 0 else 0,
            'topic_distribution': topic_counts,
            'total_quizzes': total_quizzes,
            'average_quiz_score': avg_quiz_score,
            'most_discussed_topic': max(topic_counts.items(), key=lambda x: x[1])[0] if topic_counts else None
        }

# Global instance for easy access
data_logger = DataLogger()
