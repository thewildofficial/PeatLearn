#!/usr/bin/env python3
"""
Real RAG System using Ray Peat Knowledge Base with Vector Search + LLM
Provides detailed, source-based responses with proper retrieval
"""

import asyncio
import aiohttp
import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add inference backend to path
inference_path = Path(__file__).parent.parent.parent / "inference" / "backend"
sys.path.append(str(inference_path))

try:
    from rag.vector_search import RayPeatVectorSearch, SearchResult
    from config.settings import settings
except ImportError:
    # Fallback if original modules not available
    RayPeatVectorSearch = None
    SearchResult = None
    settings = None

@dataclass
class RAGResponse:
    """Response from the RAG system."""
    answer: str
    sources: List[Any] = None
    confidence: float = 0.0
    query: str = ""

class RayPeatRAG:
    """RAG system for answering questions about Ray Peat's work using vector search + LLM."""
    
    def __init__(self, search_engine=None):
        """Initialize the RAG system."""
        self.search_engine = search_engine or (RayPeatVectorSearch() if RayPeatVectorSearch else None)
        self.llm_model = "gemini-2.5-flash"  # Updated model
        self.api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.search_engine:
            print("Warning: Vector search engine not available. Using fallback mode.")
        if not self.api_key:
            print("Warning: Google API key not found. Using fallback mode.")
        
    def get_rag_response(self, query: str, user_profile: Optional[Dict[str, Any]] = None) -> str:
        """
        Get response using proper RAG with vector search
        
        Args:
            query: User's question
            user_profile: User's learning profile for adaptation
            
        Returns:
            Detailed response with sources from Ray Peat's work
        """
        if not self.search_engine or not self.api_key:
            return self._fallback_response(query)
        
        try:
            # Use asyncio to call the async RAG system
            response = asyncio.run(self._answer_question_async(query, user_profile))
            return response.answer
        except Exception as e:
            print(f"Error in RAG system: {e}")
            return self._fallback_response(query)
    
    async def _answer_question_async(
        self, 
        question: str, 
        user_profile: Optional[Dict[str, Any]] = None,
        max_sources: int = 5,
        min_similarity: float = 0.3
    ) -> RAGResponse:
        """
        Answer a question using RAG approach with vector search.
        """
        
        # Step 1: Retrieve relevant passages using vector search
        search_results = await self.search_engine.search(
            query=question,
            top_k=max_sources,
            min_similarity=min_similarity
        )
        
        if not search_results:
            return RAGResponse(
                answer="I couldn't find any relevant information about that topic in Ray Peat's work. Please try rephrasing your question or asking about topics like metabolism, thyroid function, hormones, or nutrition.",
                sources=[],
                confidence=0.0,
                query=question
            )
        
        # Step 2: Generate answer using retrieved context
        answer = await self._generate_answer(question, search_results, user_profile)
        
        # Step 3: Calculate confidence based on similarity scores
        avg_similarity = sum(result.similarity_score for result in search_results) / len(search_results)
        confidence = min(avg_similarity * 1.2, 1.0)  # Scale similarity to confidence
        
        return RAGResponse(
            answer=answer,
            sources=search_results,
            confidence=confidence,
            query=question
        )
    
    async def _generate_answer(self, question: str, sources: List[Any], user_profile: Optional[Dict[str, Any]] = None) -> str:
        """Generate an answer using the LLM with retrieved context."""
        
        # Build context from sources
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(
                f"Source {i} (from {source.source_file}):\n"
                f"Context: {source.context}\n"
                f"Ray Peat's response: {source.ray_peat_response}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # Create adaptive prompt based on user profile
        prompt = self._create_adaptive_prompt(question, context, user_profile)

        try:
            answer = await self._call_gemini_llm(prompt)
            if answer:
                # Add source information to the answer
                source_info = f"\n\n📚 **Sources from Ray Peat's work:**\n"
                for i, source in enumerate(sources, 1):
                    source_info += f"{i}. {source.source_file} (relevance: {source.similarity_score:.2f})\n"
                
                return answer + source_info
            else:
                return "I couldn't generate a response. Please try rephrasing your question."
                
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"I encountered an error while generating the response: {str(e)}"
    
    def _create_adaptive_prompt(self, question: str, context: str, user_profile: Optional[Dict[str, Any]] = None) -> str:
        """Create a prompt adapted to the user's learning profile"""
        
        # Base prompt with retrieved context
        base_prompt = f"""You are an expert on Ray Peat's bioenergetic approach to health and biology. Answer the following question based ONLY on the provided sources from Ray Peat's work.

Question: {question}

Ray Peat's Knowledge Sources:
{context}

Instructions:
1. Answer based ONLY on the information provided in the sources above
2. If the sources don't contain enough information to answer the question, say so
3. Quote or reference specific parts of Ray Peat's responses when possible
4. Maintain Ray Peat's perspective and terminology (bioenergetic, metabolic rate, etc.)
5. Be accurate and don't make assumptions beyond what's stated
6. If multiple sources give different perspectives, acknowledge this"""
        
        if not user_profile:
            return base_prompt + "\n\nProvide a comprehensive answer with specific details from Ray Peat's work.\n\nAnswer:"
        
        # Get user's learning state and style
        learning_state = user_profile.get('overall_state', 'learning')
        learning_style = user_profile.get('learning_style', 'balanced')
        
        # Adapt complexity based on learning state
        if learning_state == 'struggling':
            complexity_instruction = """
7. Keep your answer simple and foundational:
   - Focus on basic concepts and principles from the sources
   - Use clear, accessible language
   - Provide practical, actionable advice from Ray Peat
   - Avoid overly technical biochemical details unless necessary
            """
        elif learning_state == 'advanced':
            complexity_instruction = """
7. Provide an advanced, detailed response:
   - Include specific biochemical mechanisms mentioned in the sources
   - Reference metabolic pathways and cellular processes Ray Peat discusses
   - Discuss nuances and exceptions to general principles
   - Connect to broader physiological systems as described by Ray Peat
            """
        else:  # learning
            complexity_instruction = """
7. Provide a balanced response:
   - Explain key concepts clearly from the sources
   - Include some technical details with explanations
   - Balance theory with practical applications Ray Peat mentions
   - Build on foundational knowledge
            """
        
        # Adapt style based on learning preference
        if learning_style == 'explorer':
            style_instruction = """
8. Structure your response to encourage exploration:
   - Mention related topics and connections Ray Peat makes
   - Suggest areas for further investigation based on the sources
   - Show how this topic relates to other aspects of health in Ray Peat's work
            """
        elif learning_style == 'deep_diver':
            style_instruction = """
8. Provide deep, focused information:
   - Go into detailed mechanisms and explanations from the sources
   - Provide thorough coverage of the specific topic
   - Include relevant research and scientific backing Ray Peat mentions
            """
        else:  # balanced
            style_instruction = """
8. Provide a well-rounded response:
   - Balance depth with breadth based on available sources
   - Include both theoretical understanding and practical applications Ray Peat discusses
            """
        
        return f"""{base_prompt}
        
{complexity_instruction}

{style_instruction}

Answer:"""
    
    async def _call_gemini_llm(self, prompt: str) -> Optional[str]:
        """Call Gemini LLM for text generation."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.llm_model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.3,  # Lower temperature for more factual responses
                "maxOutputTokens": 1500,  # Increased for more detailed responses
                "topP": 0.8,
                "topK": 40
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "candidates" in result and len(result["candidates"]) > 0:
                            return result["candidates"][0]["content"]["parts"][0]["text"]
                        else:
                            return None
                    else:
                        error_text = await response.text()
                        print(f"LLM API Error {response.status}: {error_text}")
                        return None
                        
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return None
    
    def _fallback_response(self, query: str) -> str:
        """Provide a fallback response when full RAG is unavailable"""
        return f"""I understand you're asking about: "{query}"

I'm currently unable to access my full Ray Peat knowledge base with vector search, but here are some general Ray Peat principles that might be relevant:

🔋 **Bioenergetic Approach:**
- Focus on supporting cellular energy production (ATP)
- Optimize thyroid function and metabolic rate
- Balance stress hormones (cortisol) with protective hormones (progesterone)

🍯 **Nutritional Principles:**
- Simple sugars (honey, fruit) for quick energy
- Saturated fats for hormone production
- Avoid polyunsaturated fats (PUFA) which can be inflammatory
- Adequate protein for tissue repair

☀️ **Environmental Factors:**
- Adequate light exposure (especially morning sunlight)
- Maintain warm body temperature
- Minimize stress and support recovery

⚠️ **Note:** This is a fallback response. For detailed answers with specific sources from Ray Peat's work, the vector search system needs to be properly configured with the Ray Peat corpus.

To get better responses, please ensure:
1. The embedding vectors are available in `embedding/vectors/`
2. The processed Ray Peat corpus is available in `data/processed/`
3. The vector search system is properly initialized

For more specific information, I recommend consulting Ray Peat's newsletters, articles, or books directly.
        """

# Global instance
ray_peat_rag = RayPeatRAG()