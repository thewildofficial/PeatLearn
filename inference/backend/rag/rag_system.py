#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) System for Ray Peat Knowledge

Combines vector search with LLM generation for accurate Q&A.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .vector_search import RayPeatVectorSearch, SearchResult
from config.settings import settings

@dataclass
class RAGResponse:
    """Response from the RAG system."""
    answer: str
    sources: List[SearchResult]
    confidence: float
    query: str

class RayPeatRAG:
    """RAG system for answering questions about Ray Peat's work."""
    
    def __init__(self, search_engine: RayPeatVectorSearch = None):
        """Initialize the RAG system."""
        self.search_engine = search_engine or RayPeatVectorSearch()
        self.llm_model = "gemini-2.5-flash"  # Fast and cost-effective for RAG
        
    async def answer_question(
        self, 
        question: str, 
        max_sources: int = 5,
        min_similarity: float = 0.3
    ) -> RAGResponse:
        """
        Answer a question using RAG approach.
        
        Args:
            question: The user's question
            max_sources: Maximum number of source passages to use
            min_similarity: Minimum similarity threshold for sources
            
        Returns:
            RAGResponse with answer and sources
        """
        
        # Step 1: Retrieve relevant passages
        search_results = await self.search_engine.search(
            query=question,
            top_k=max_sources,
            min_similarity=min_similarity
        )
        
        if not search_results:
            return RAGResponse(
                answer="I couldn't find any relevant information about that topic in Ray Peat's work.",
                sources=[],
                confidence=0.0,
                query=question
            )
        
        # Step 2: Generate answer using retrieved context
        answer = await self._generate_answer(question, search_results)
        
        # Step 3: Calculate confidence based on similarity scores
        avg_similarity = sum(result.similarity_score for result in search_results) / len(search_results)
        confidence = min(avg_similarity * 1.2, 1.0)  # Scale similarity to confidence
        
        return RAGResponse(
            answer=answer,
            sources=search_results,
            confidence=confidence,
            query=question
        )
    
    async def _generate_answer(self, question: str, sources: List[SearchResult]) -> str:
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
        
        # Create the prompt
        prompt = f"""You are an expert on Ray Peat's bioenergetic approach to health and biology. Answer the following question based ONLY on the provided sources from Ray Peat's work.

Question: {question}

Ray Peat's Knowledge Sources:
{context}

Instructions:
1. Answer based ONLY on the information provided in the sources above
2. If the sources don't contain enough information to answer the question, say so
3. Quote or reference specific parts of Ray Peat's responses when possible
4. Maintain Ray Peat's perspective and terminology (bioenergetic, metabolic rate, etc.)
5. Be accurate and don't make assumptions beyond what's stated
6. If multiple sources give different perspectives, acknowledge this

Answer:"""

        try:
            answer = await self._call_gemini_llm(prompt)
            return answer or "I couldn't generate a response. Please try rephrasing your question."
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"I encountered an error while generating the response: {str(e)}"
    
    async def _call_gemini_llm(self, prompt: str) -> Optional[str]:
        """Call Gemini LLM for text generation."""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.llm_model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": settings.GEMINI_API_KEY
        }
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.3,  # Lower temperature for more factual responses
                "maxOutputTokens": 1000,
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
    
    async def get_related_topics(self, query: str, max_topics: int = 8) -> List[str]:
        """Get related topics based on the query."""
        search_results = await self.search_engine.search(
            query=query,
            top_k=max_topics * 2,
            min_similarity=0.2
        )
        
        # Extract unique topics from contexts
        topics = set()
        for result in search_results:
            # Simple topic extraction from context
            context_words = result.context.lower().split()
            # Look for key topic indicators
            for word in context_words:
                if len(word) > 4 and word not in ['about', 'discusses', 'mentions', 'talks']:
                    topics.add(word.title())
        
        return list(topics)[:max_topics]

# Global instance for the API
rag_system = RayPeatRAG()
