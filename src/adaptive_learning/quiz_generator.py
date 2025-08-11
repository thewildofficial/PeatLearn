#!/usr/bin/env python3
"""
Quiz Generator for PeatLearn Adaptive Learning System
Creates personalized quizzes based on user profiles and recent topics
"""

import json
import random
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from .profile_analyzer import TopicExtractor

class QuizGenerator:
    """
    Generates adaptive quizzes based on user learning profiles and topics
    """
    
    def __init__(self, profiler=None):
        self.topic_extractor = TopicExtractor()
        self.profiler = profiler
        
        # Quiz question templates organized by topic and difficulty
        self.question_templates = {
            'metabolism': {
                'beginner': [
                    {
                        'question': 'What is the main function of thyroid hormones in the body?',
                        'options': ['Regulate heart rate only', 'Control metabolism and energy production', 'Digest food', 'Filter blood'],
                        'correct': 1,
                        'explanation': 'Thyroid hormones, particularly T3, are the main regulators of cellular metabolism and energy production throughout the body.'
                    },
                    {
                        'question': 'According to Ray Peat, what is a sign of good metabolic health?',
                        'options': ['Feeling cold all the time', 'Having warm hands and feet', 'Sleeping 12+ hours', 'Constant hunger'],
                        'correct': 1,
                        'explanation': 'Ray Peat emphasized that good metabolic health is indicated by warm extremities, showing good circulation and energy production.'
                    }
                ],
                'intermediate': [
                    {
                        'question': 'What does Ray Peat say about the relationship between CO2 and metabolism?',
                        'options': ['CO2 is just waste', 'CO2 is essential for cellular function', 'CO2 causes acidosis', 'CO2 should be avoided'],
                        'correct': 1,
                        'explanation': 'Ray Peat taught that CO2 is not just waste but an essential molecule that supports cellular respiration and energy production.'
                    },
                    {
                        'question': 'Which process does Ray Peat believe is superior for energy production?',
                        'options': ['Glycolysis', 'Oxidative phosphorylation', 'Fermentation', 'Ketosis'],
                        'correct': 1,
                        'explanation': 'Ray Peat emphasized that oxidative phosphorylation (using oxygen) is the most efficient and healthy way to produce cellular energy.'
                    }
                ],
                'advanced': [
                    {
                        'question': 'How does Ray Peat view the Warburg effect in relation to health?',
                        'options': ['It\'s always beneficial', 'It\'s a sign of cellular dysfunction', 'It only occurs in cancer', 'It\'s unrelated to health'],
                        'correct': 1,
                        'explanation': 'Ray Peat viewed the Warburg effect (cellular reliance on glycolysis even with oxygen present) as a sign of mitochondrial dysfunction and poor metabolic health.'
                    }
                ]
            },
            'hormones': {
                'beginner': [
                    {
                        'question': 'What is progesterone\'s main role according to Ray Peat?',
                        'options': ['Only for pregnancy', 'Protective anti-stress hormone', 'Causes inflammation', 'Increases estrogen'],
                        'correct': 1,
                        'explanation': 'Ray Peat viewed progesterone as a key protective hormone that opposes stress and supports healthy metabolism.'
                    },
                    {
                        'question': 'How did Ray Peat view estrogen in excess?',
                        'options': ['Always beneficial', 'Can be problematic and inflammatory', 'Only affects reproduction', 'Has no metabolic effects'],
                        'correct': 1,
                        'explanation': 'Ray Peat believed that excess estrogen (estrogen dominance) could be inflammatory and disruptive to healthy metabolism.'
                    }
                ],
                'intermediate': [
                    {
                        'question': 'What did Ray Peat say about cortisol\'s effects?',
                        'options': ['Always harmful', 'Protective in acute stress, harmful when chronic', 'Only affects mood', 'Improves metabolism'],
                        'correct': 1,
                        'explanation': 'Ray Peat recognized that while cortisol can be protective during acute stress, chronic elevation becomes harmful and metabolically disruptive.'
                    }
                ],
                'advanced': [
                    {
                        'question': 'How does Ray Peat explain the pregnenolone steal?',
                        'options': ['Stress diverts pregnenolone to cortisol production', 'Pregnenolone is destroyed by inflammation', 'It only occurs in women', 'It\'s a myth'],
                        'correct': 0,
                        'explanation': 'Ray Peat described how chronic stress can divert pregnenolone (the hormone precursor) toward cortisol production at the expense of other protective hormones like progesterone.'
                    }
                ]
            },
            'nutrition': {
                'beginner': [
                    {
                        'question': 'What type of sugar did Ray Peat generally recommend?',
                        'options': ['High fructose corn syrup', 'Table sugar (sucrose)', 'Artificial sweeteners', 'No sugar at all'],
                        'correct': 1,
                        'explanation': 'Ray Peat generally favored sucrose (table sugar) over high fructose corn syrup, believing it was metabolically easier to handle.'
                    },
                    {
                        'question': 'Which food did Ray Peat often recommend for its nutritional completeness?',
                        'options': ['Kale', 'Milk', 'Quinoa', 'Chicken breast'],
                        'correct': 1,
                        'explanation': 'Ray Peat frequently recommended milk for its complete amino acid profile, calcium, and overall nutritional density.'
                    }
                ],
                'intermediate': [
                    {
                        'question': 'What did Ray Peat say about polyunsaturated fats (PUFAs)?',
                        'options': ['They\'re essential and beneficial', 'They can be inflammatory and disruptive', 'They only affect cholesterol', 'They\'re the best energy source'],
                        'correct': 1,
                        'explanation': 'Ray Peat was critical of excess polyunsaturated fats, believing they could be inflammatory and interfere with healthy metabolism.'
                    }
                ],
                'advanced': [
                    {
                        'question': 'How did Ray Peat explain the role of gelatin in nutrition?',
                        'options': ['Just for joint health', 'Provides methionine balance and glycine', 'Only for muscle building', 'Has no special benefits'],
                        'correct': 1,
                        'explanation': 'Ray Peat valued gelatin for providing glycine to balance methionine from muscle meats, supporting healthy protein metabolism.'
                    }
                ]
            },
            'stress': {
                'beginner': [
                    {
                        'question': 'According to Ray Peat, what happens to metabolism during chronic stress?',
                        'options': ['It improves', 'It becomes less efficient', 'It stays the same', 'It only affects mood'],
                        'correct': 1,
                        'explanation': 'Ray Peat taught that chronic stress impairs metabolic efficiency and shifts the body toward less optimal energy production.'
                    }
                ],
                'intermediate': [
                    {
                        'question': 'What did Ray Peat say about the relationship between stress and inflammation?',
                        'options': ['They\'re unrelated', 'Stress promotes inflammation', 'Inflammation prevents stress', 'They oppose each other'],
                        'correct': 1,
                        'explanation': 'Ray Peat explained how chronic stress promotes inflammatory processes in the body, creating a cycle of dysfunction.'
                    }
                ],
                'advanced': [
                    {
                        'question': 'How did Ray Peat connect stress to serotonin?',
                        'options': ['Serotonin reduces stress', 'Stress increases beneficial serotonin', 'Stress can increase problematic serotonin', 'They\'re unconnected'],
                        'correct': 2,
                        'explanation': 'Ray Peat believed that stress could lead to increased serotonin in tissues where it becomes inflammatory and metabolically disruptive.'
                    }
                ]
            }
        }
    
    def generate_quiz(self, 
                     user_profile: Dict[str, Any],
                     topic: Optional[str] = None,
                     num_questions: int = 5,
                     recent_interactions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a personalized quiz based on user profile
        
        Args:
            user_profile: User's learning profile
            topic: Specific topic to focus on (optional)
            num_questions: Number of questions to generate
            recent_interactions: Recent user interactions for context
            
        Returns:
            Quiz dictionary with questions and metadata
        """
        
        # Determine quiz topic and difficulty
        quiz_topic, difficulty_level = self._determine_quiz_parameters(
            user_profile, topic, recent_interactions
        )
        
        # Generate questions
        questions = self._generate_questions(quiz_topic, difficulty_level, num_questions)
        
        # Create quiz metadata
        quiz_id = str(uuid.uuid4())
        quiz_metadata = {
            'quiz_id': quiz_id,
            'topic': quiz_topic,
            'difficulty': difficulty_level,
            'num_questions': len(questions),
            'created_at': datetime.now().isoformat(),
            'user_id': user_profile.get('user_id', 'unknown'),
            'adaptation_info': {
                'user_state': user_profile.get('overall_state', 'learning'),
                'topic_mastery': user_profile.get('topic_mastery', {}).get(quiz_topic, {}),
                'learning_style': user_profile.get('learning_style', 'explorer')
            }
        }
        
        return {
            'quiz_id': quiz_id,
            'quiz_metadata': quiz_metadata,
            'questions': questions,
            'instructions': self._get_quiz_instructions(difficulty_level, user_profile),
            'estimated_time_minutes': len(questions) * 1.5  # Estimate 1.5 minutes per question
        }
    
    def _determine_quiz_parameters(self, 
                                  user_profile: Dict[str, Any],
                                  requested_topic: Optional[str],
                                  recent_interactions: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Determine quiz topic and difficulty based on user profile
        
        Args:
            user_profile: User's learning profile
            requested_topic: Specifically requested topic
            recent_interactions: Recent user interactions
            
        Returns:
            Tuple of (topic, difficulty_level)
        """
        
        # Determine topic
        if requested_topic and requested_topic in self.question_templates:
            topic = requested_topic
        else:
            # Choose topic based on recent interactions or profile
            topic = self._choose_topic_from_profile(user_profile, recent_interactions)
        
        # Determine difficulty level
        difficulty = self._determine_difficulty_level(user_profile, topic)
        
        return topic, difficulty
    
    def _choose_topic_from_profile(self, 
                                  user_profile: Dict[str, Any],
                                  recent_interactions: List[Dict[str, Any]]) -> str:
        """
        Choose quiz topic based on user profile and recent activity
        
        Args:
            user_profile: User's learning profile
            recent_interactions: Recent user interactions
            
        Returns:
            Selected topic name
        """
        
        topic_mastery = user_profile.get('topic_mastery', {})
        learning_style = user_profile.get('learning_style', 'explorer')
        
        # Analyze recent interactions for topic frequency
        recent_topics = []
        if recent_interactions:
            for interaction in recent_interactions[-10:]:  # Last 10 interactions
                query = interaction.get('user_query', '')
                primary_topic = self.topic_extractor.get_primary_topic(query)
                if primary_topic:
                    recent_topics.append(primary_topic)
        
        # Choose topic based on learning style and mastery
        available_topics = list(self.question_templates.keys())
        
        if learning_style == 'deep_diver' and recent_topics:
            # Focus on most recent topic for deep divers
            most_recent_topic = recent_topics[-1] if recent_topics else None
            if most_recent_topic and most_recent_topic in available_topics:
                return most_recent_topic
        
        elif learning_style == 'explorer':
            # Choose topic they haven't mastered yet for explorers
            for topic in available_topics:
                if topic not in topic_mastery or topic_mastery[topic]['state'] != 'advanced':
                    return topic
        
        # Default: choose topic with lowest mastery or random if none
        if topic_mastery:
            lowest_mastery_topic = min(
                topic_mastery.items(),
                key=lambda x: x[1].get('mastery_level', 0)
            )[0]
            if lowest_mastery_topic in available_topics:
                return lowest_mastery_topic
        
        # Fallback: random topic
        return random.choice(available_topics)
    
    def _determine_difficulty_level(self, user_profile: Dict[str, Any], topic: str) -> str:
        """
        Determine appropriate difficulty level for the quiz
        
        Args:
            user_profile: User's learning profile
            topic: Quiz topic
            
        Returns:
            Difficulty level: 'beginner', 'intermediate', or 'advanced'
        """
        
        overall_state = user_profile.get('overall_state', 'learning')
        topic_mastery = user_profile.get('topic_mastery', {})
        total_interactions = user_profile.get('total_interactions', 0)
        
        # Check topic-specific mastery
        if topic in topic_mastery:
            topic_state = topic_mastery[topic]['state']
            mastery_level = topic_mastery[topic]['mastery_level']
            
            if topic_state == 'struggling' or mastery_level < 0.4:
                return 'beginner'
            elif topic_state == 'advanced' or mastery_level > 0.8:
                return 'advanced'
            else:
                return 'intermediate'
        
        # Use overall state if no topic-specific data
        if overall_state == 'struggling' or total_interactions < 5:
            return 'beginner'
        elif overall_state == 'advanced' and total_interactions > 15:
            return 'advanced'
        else:
            return 'intermediate'
    
    def _generate_questions(self, topic: str, difficulty: str, num_questions: int) -> List[Dict[str, Any]]:
        """
        Generate quiz questions for the specified topic and difficulty
        
        Args:
            topic: Topic name
            difficulty: Difficulty level
            num_questions: Number of questions to generate
            
        Returns:
            List of question dictionaries
        """
        
        questions = []
        
        # Get available questions for topic and difficulty
        available_questions = self.question_templates.get(topic, {}).get(difficulty, [])
        
        if not available_questions:
            # Fallback to beginner level if requested difficulty not available
            available_questions = self.question_templates.get(topic, {}).get('beginner', [])
        
        if not available_questions:
            # Ultimate fallback: get questions from any topic
            for fallback_topic in self.question_templates:
                available_questions = self.question_templates[fallback_topic].get(difficulty, [])
                if available_questions:
                    break
        
        # Select questions (with replacement if needed)
        selected_questions = []
        question_pool = available_questions.copy()
        
        for i in range(num_questions):
            if not question_pool:
                # Refill pool if we've used all questions
                question_pool = available_questions.copy()
            
            question_template = random.choice(question_pool)
            question_pool.remove(question_template)
            
            # Create question with unique ID
            question = question_template.copy()
            question['question_id'] = f"{topic}_{difficulty}_{i+1}_{random.randint(1000, 9999)}"
            question['topic'] = topic
            question['difficulty_level'] = difficulty
            
            selected_questions.append(question)
        
        return selected_questions
    
    def _get_quiz_instructions(self, difficulty: str, user_profile: Dict[str, Any]) -> str:
        """
        Get personalized quiz instructions
        
        Args:
            difficulty: Quiz difficulty level
            user_profile: User's learning profile
            
        Returns:
            Instructions string
        """
        
        base_instructions = "Welcome to your personalized Ray Peat quiz! "
        
        if difficulty == 'beginner':
            base_instructions += "This quiz covers fundamental concepts. Take your time and don't worry if some questions seem challenging - that's how we learn!"
        
        elif difficulty == 'intermediate':
            base_instructions += "This quiz explores Ray Peat's ideas in more depth. You should be familiar with basic concepts."
        
        elif difficulty == 'advanced':
            base_instructions += "This is an advanced quiz covering complex bioenergetic concepts. Good luck!"
        
        # Add personalized note based on learning style
        learning_style = user_profile.get('learning_style', 'explorer')
        
        if learning_style == 'deep_diver':
            base_instructions += " Since you like to dive deep into topics, these questions will help reinforce your detailed understanding."
        
        elif learning_style == 'explorer':
            base_instructions += " These questions will help you connect different concepts across Ray Peat's work."
        
        base_instructions += "\n\nRemember: This quiz is a learning tool. Each question includes an explanation to help you understand the concepts better!"
        
        return base_instructions
    
    def evaluate_quiz(self, quiz_data: Dict[str, Any], user_answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate quiz results and provide feedback
        
        Args:
            quiz_data: Original quiz data
            user_answers: User's answers (question_id -> selected_index)
            
        Returns:
            Evaluation results with score and feedback
        """
        
        questions = quiz_data['questions']
        quiz_metadata = quiz_data['quiz_metadata']
        
        total_questions = len(questions)
        correct_answers = 0
        question_results = []
        
        # Evaluate each question
        for question in questions:
            question_id = question['question_id']
            correct_index = question['correct']
            user_index = user_answers.get(question_id, -1)
            
            is_correct = user_index == correct_index
            if is_correct:
                correct_answers += 1
            
            question_results.append({
                'question_id': question_id,
                'question': question['question'],
                'user_answer': question['options'][user_index] if 0 <= user_index < len(question['options']) else 'No answer',
                'correct_answer': question['options'][correct_index],
                'is_correct': is_correct,
                'explanation': question['explanation']
            })
        
        # Calculate score
        score_percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        # Generate feedback
        feedback = self._generate_quiz_feedback(score_percentage, quiz_metadata, question_results)
        
        return {
            'quiz_id': quiz_data['quiz_id'],
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'score_percentage': score_percentage,
            'feedback': feedback,
            'question_results': question_results,
            'completed_at': datetime.now().isoformat()
        }
    
    def _generate_quiz_feedback(self, 
                               score_percentage: float,
                               quiz_metadata: Dict[str, Any],
                               question_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate personalized feedback based on quiz performance
        
        Args:
            score_percentage: Quiz score as percentage
            quiz_metadata: Quiz metadata
            question_results: Results for each question
            
        Returns:
            Feedback dictionary
        """
        
        feedback = {}
        
        # Overall performance feedback
        if score_percentage >= 90:
            feedback['overall'] = "Excellent work! You have a strong understanding of Ray Peat's concepts. 🌟"
            feedback['level'] = 'excellent'
        elif score_percentage >= 70:
            feedback['overall'] = "Good job! You're well on your way to mastering these concepts. 👍"
            feedback['level'] = 'good'
        elif score_percentage >= 50:
            feedback['overall'] = "You're making progress! Keep studying and asking questions. 📚"
            feedback['level'] = 'developing'
        else:
            feedback['overall'] = "This is challenging material - don't get discouraged! Focus on the fundamentals. 💪"
            feedback['level'] = 'needs_work'
        
        # Specific recommendations
        difficulty = quiz_metadata.get('difficulty', 'intermediate')
        topic = quiz_metadata.get('topic', 'general')
        
        recommendations = []
        
        if feedback['level'] in ['needs_work', 'developing']:
            recommendations.append(f"Review the basics of {topic} before moving to more advanced concepts")
            recommendations.append("Don't hesitate to ask for simpler explanations")
            recommendations.append("Focus on understanding one concept at a time")
        
        elif feedback['level'] == 'good':
            recommendations.append(f"You're ready for more advanced {topic} concepts")
            recommendations.append("Try exploring how this topic connects to other areas")
        
        elif feedback['level'] == 'excellent':
            if difficulty != 'advanced':
                recommendations.append("Consider trying more advanced quizzes")
            recommendations.append(f"Explore the nuanced aspects of {topic}")
            recommendations.append("Help deepen your understanding by teaching others")
        
        feedback['recommendations'] = recommendations
        
        # Areas for improvement
        incorrect_questions = [q for q in question_results if not q['is_correct']]
        if incorrect_questions:
            feedback['areas_for_improvement'] = [
                f"Review: {q['question']}" for q in incorrect_questions[:3]  # Top 3
            ]
        
        return feedback

# Global instance for easy access
quiz_generator = QuizGenerator()
