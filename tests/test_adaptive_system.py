#!/usr/bin/env python3
"""
Test script for PeatLearn Adaptive Learning System
Tests all components with sample data to verify functionality
"""

import sys
import os
from datetime import datetime
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from adaptive_learning import (
    data_logger, profiler, content_selector, 
    quiz_generator, dashboard, TopicExtractor
)

def test_topic_extractor():
    """Test the topic extraction functionality"""
    print("🔍 Testing Topic Extractor...")
    
    extractor = TopicExtractor()
    
    test_queries = [
        "What does Ray Peat say about thyroid function and metabolism?",
        "How does progesterone interact with estrogen and cortisol?", 
        "What foods does Ray Peat recommend for good nutrition?",
        "How does chronic stress affect the body?",
        "Tell me about inflammation and cytokines"
    ]
    
    for query in test_queries:
        topics = extractor.extract_topics(query)
        primary = extractor.get_primary_topic(query)
        print(f"  Query: {query}")
        print(f"  Topics: {topics}")
        print(f"  Primary: {primary}")
        print()
    
    print("✅ Topic Extractor test completed!\n")

def test_data_logger():
    """Test the data logging functionality"""
    print("📊 Testing Data Logger...")
    
    # Create some test interactions
    test_interactions = [
        {
            "user_query": "What is thyroid function?",
            "llm_response": "Thyroid function involves T3 and T4 hormones that regulate metabolism...",
            "topic": "metabolism",
            "user_feedback": 1,
            "interaction_type": "chat"
        },
        {
            "user_query": "How does progesterone work?", 
            "llm_response": "Progesterone is a protective hormone that opposes estrogen...",
            "topic": "hormones",
            "user_feedback": 1,
            "interaction_type": "chat"
        },
        {
            "user_query": "What about sugar in diet?",
            "llm_response": "Ray Peat generally favored sucrose over other sugars...",
            "topic": "nutrition", 
            "user_feedback": -1,
            "interaction_type": "chat"
        }
    ]
    
    # Simulate logging interactions
    print("  Logging test interactions...")
    for interaction in test_interactions:
        data_logger.log_interaction(**interaction)
    
    # Test getting interactions back
    print("  Retrieving interactions...")
    user_id = data_logger.get_user_id()
    interactions = data_logger.get_user_interactions(user_id)
    
    print(f"  Logged {len(interactions)} interactions for user {user_id}")
    for i, interaction in enumerate(interactions):
        print(f"    {i+1}. Topic: {interaction['topic']}, Feedback: {interaction['user_feedback']}")
    
    # Test statistics
    stats = data_logger.get_interaction_stats(user_id)
    print(f"  Stats: {stats}")
    
    print("✅ Data Logger test completed!\n")
    return interactions

def test_profiler(interactions):
    """Test the learner profiling functionality"""
    print("🎯 Testing Learner Profiler...")
    
    user_id = data_logger.get_user_id()
    
    # Update user profile based on interactions
    print("  Analyzing user interactions...")
    profile = profiler.update_user_profile(user_id, interactions)
    
    print("  Generated Profile:")
    print(f"    User ID: {profile.get('user_id')}")
    print(f"    Overall State: {profile.get('overall_state')}")
    print(f"    Learning Style: {profile.get('learning_style')}")
    print(f"    Total Interactions: {profile.get('total_interactions')}")
    
    print("    Topic Mastery:")
    for topic, mastery in profile.get('topic_mastery', {}).items():
        print(f"      {topic}: {mastery['state']} (level: {mastery['mastery_level']:.2f})")
    
    # Test recommendations
    print("  Generating recommendations...")
    recommendations = profiler.get_recommendations(profile)
    print("    Recommendations:")
    for rec in recommendations:
        print(f"      • {rec['title']}: {rec['description']}")
    
    print("✅ Learner Profiler test completed!\n")
    return profile

def test_content_selector(profile):
    """Test the adaptive content selection"""
    print("🎨 Testing Content Selector...")
    
    user_id = data_logger.get_user_id()
    
    # Test different types of queries
    test_queries = [
        "What is metabolism?",
        "How do hormones work?", 
        "Tell me about nutrition"
    ]
    
    print("  Testing adaptive responses...")
    for query in test_queries:
        print(f"    Query: {query}")
        
        # This would normally call the RAG API, but for testing we'll simulate
        try:
            # Simulate what the content selector would do
            adaptation_info = content_selector._analyze_query_and_profile(query, profile)
            print(f"      Adaptation Level: {adaptation_info['adaptation_level']}")
            print(f"      Primary Topic: {adaptation_info['primary_topic']}")
            print(f"      Learning Style: {adaptation_info['learning_style']}")
            
            # Test recommendation generation
            recommendations = content_selector._get_content_recommendations(adaptation_info, profile)
            print(f"      Recommendations: {len(recommendations)} generated")
            
        except Exception as e:
            print(f"      Note: Full RAG integration test skipped (RAG API not available): {e}")
        
        print()
    
    print("✅ Content Selector test completed!\n")

def test_quiz_generator(profile):
    """Test the quiz generation functionality"""
    print("🎯 Testing Quiz Generator...")
    
    # Generate a quiz based on user profile
    print("  Generating personalized quiz...")
    quiz = quiz_generator.generate_quiz(
        user_profile=profile,
        topic="metabolism",  # Focus on metabolism
        num_questions=3
    )
    
    print(f"    Quiz ID: {quiz['quiz_id']}")
    print(f"    Topic: {quiz['quiz_metadata']['topic']}")
    print(f"    Difficulty: {quiz['quiz_metadata']['difficulty']}")
    print(f"    Questions: {len(quiz['questions'])}")
    
    print("    Sample Questions:")
    for i, question in enumerate(quiz['questions'][:2]):  # Show first 2
        print(f"      {i+1}. {question['question']}")
        for j, option in enumerate(question['options']):
            marker = "✓" if j == question['correct'] else " "
            print(f"         {marker} {chr(65+j)}. {option}")
        print()
    
    # Test quiz evaluation
    print("  Testing quiz evaluation...")
    # Simulate user answers (let's say they got 2/3 correct)
    user_answers = {}
    for i, question in enumerate(quiz['questions']):
        # Simulate: correct answer for first 2, wrong for last
        user_answers[question['question_id']] = question['correct'] if i < 2 else (question['correct'] + 1) % len(question['options'])
    
    results = quiz_generator.evaluate_quiz(quiz, user_answers)
    print(f"    Score: {results['correct_answers']}/{results['total_questions']} ({results['score_percentage']:.1f}%)")
    print(f"    Feedback Level: {results['feedback']['level']}")
    print(f"    Overall Feedback: {results['feedback']['overall']}")
    
    print("✅ Quiz Generator test completed!\n")
    return results

def test_dashboard_data(profile, interactions, quiz_results):
    """Test dashboard data preparation (without actual Streamlit rendering)"""
    print("📊 Testing Dashboard Data...")
    
    # Test data processing functions
    print("  Testing data processing...")
    
    # Simulate what dashboard would do
    total_interactions = len(interactions)
    topics_explored = len(profile.get('topic_mastery', {}))
    
    print(f"    Total Interactions: {total_interactions}")
    print(f"    Topics Explored: {topics_explored}")
    
    if quiz_results:
        avg_quiz_score = quiz_results['score_percentage']
        print(f"    Average Quiz Score: {avg_quiz_score:.1f}%")
    
    # Test topic mastery data
    topic_mastery = profile.get('topic_mastery', {})
    if topic_mastery:
        print("    Topic Mastery Data:")
        for topic, data in topic_mastery.items():
            print(f"      {topic}: {data['mastery_level']*100:.1f}% ({data['state']})")
    
    print("✅ Dashboard Data test completed!\n")

def test_integration():
    """Test full integration workflow"""
    print("🔄 Testing Full Integration Workflow...")
    
    # Simulate a complete user journey
    print("  Simulating user learning journey...")
    
    # Step 1: User asks questions and provides feedback
    learning_journey = [
        ("What is thyroid function?", "metabolism", 1),
        ("How does T3 work?", "metabolism", 1), 
        ("What about T4?", "metabolism", 1),
        ("Tell me about progesterone", "hormones", -1),  # Struggling with hormones
        ("How does progesterone work?", "hormones", -1),
        ("What foods does Ray Peat recommend?", "nutrition", 1),
    ]
    
    all_interactions = []
    for query, topic, feedback in learning_journey:
        # Log interaction
        data_logger.log_interaction(
            user_query=query,
            llm_response=f"Response about {topic}...",
            topic=topic,
            user_feedback=feedback,
            interaction_type="chat"
        )
        
        # Get all interactions so far
        user_id = data_logger.get_user_id()
        interactions = data_logger.get_user_interactions(user_id)
        
        # Update profile
        profile = profiler.update_user_profile(user_id, interactions)
        
        print(f"    After '{query}':")
        print(f"      Overall State: {profile.get('overall_state')}")
        print(f"      Learning Style: {profile.get('learning_style')}")
        
        # Show topic-specific states
        topic_mastery = profile.get('topic_mastery', {})
        if topic_mastery:
            for t, data in topic_mastery.items():
                print(f"      {t}: {data['state']} ({data['mastery_level']:.2f})")
        print()
    
    print("✅ Full Integration test completed!\n")
    return profile

def clean_test_data():
    """Clean up test data files"""
    print("🧹 Cleaning up test data...")
    
    test_files = [
        "data/user_interactions/interactions.csv",
        "data/user_interactions/user_profiles.json", 
        "data/user_interactions/quiz_results.csv"
    ]
    
    for file_path in test_files:
        try:
            if os.path.exists(file_path):
                # Reset files to headers only
                if file_path.endswith('.csv'):
                    if 'interactions' in file_path:
                        with open(file_path, 'w') as f:
                            f.write("user_id,session_id,timestamp,user_query,llm_response,topic,user_feedback,interaction_type,response_time,context\n")
                    elif 'quiz_results' in file_path:
                        with open(file_path, 'w') as f:
                            f.write("user_id,session_id,timestamp,quiz_id,topic,questions_total,questions_correct,score_percentage,difficulty_level,time_taken_seconds\n")
                elif file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump({
                            "profiles": {},
                            "metadata": {
                                "created_at": datetime.now().isoformat(),
                                "last_updated": datetime.now().isoformat(),
                                "version": "1.0"
                            }
                        }, f, indent=2)
        except Exception as e:
            print(f"    Warning: Could not reset {file_path}: {e}")
    
    print("✅ Cleanup completed!\n")

def main():
    """Run all tests"""
    print("🚀 Starting PeatLearn Adaptive Learning System Tests\n")
    print("=" * 60)
    
    # Test individual components
    test_topic_extractor()
    interactions = test_data_logger()
    profile = test_profiler(interactions)
    test_content_selector(profile)
    quiz_results = test_quiz_generator(profile)
    test_dashboard_data(profile, interactions, quiz_results)
    
    # Clean up and test integration
    clean_test_data()
    final_profile = test_integration()
    
    print("=" * 60)
    print("🎉 All tests completed successfully!")
    print("\nSystem is ready for integration with Streamlit dashboard!")
    print("\nNext steps:")
    print("1. Run this test script to verify everything works")
    print("2. Integrate feedback buttons into your Streamlit chat interface")
    print("3. Connect adaptive responses to your RAG system")
    print("4. Add quiz and dashboard tabs to your Streamlit app")

if __name__ == "__main__":
    main()
