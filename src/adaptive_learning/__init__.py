#!/usr/bin/env python3
"""
PeatLearn Adaptive Learning System

This package provides adaptive learning capabilities for the PeatLearn platform,
including user profiling, content adaptation, quiz generation, and progress tracking.

Main Components:
- DataLogger: Tracks user interactions and feedback
- LearnerProfiler: Analyzes user behavior and creates learning profiles
- ContentSelector: Modifies RAG responses based on user profiles
- QuizGenerator: Creates personalized quizzes
- Dashboard: Visualizes learning progress and insights
"""

__version__ = "1.0.0"
__author__ = "PeatLearn Development Team"

# Import main classes for easy access
from .data_logger import DataLogger
from .profile_analyzer import LearnerProfiler, TopicExtractor
from .content_selector import ContentSelector
from .quiz_generator import QuizGenerator
from .dashboard import Dashboard

__all__ = [
    'DataLogger',
    'LearnerProfiler', 
    'TopicExtractor',
    'ContentSelector',
    'QuizGenerator',
    'Dashboard'
]
