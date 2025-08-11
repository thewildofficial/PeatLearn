#!/usr/bin/env python3
"""
Pinecone-based RAG (Retrieval-Augmented Generation) System for Ray Peat Knowledge

Combines Pinecone vector search with LLM generation for accurate Q&A.
"""

import sys
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from .vector_search import PineconeVectorSearch, SearchResult
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Response from the Pinecone-based RAG system."""
    answer: str
    sources: List[SearchResult]
    confidence: float
    query: str
    search_stats: Dict[str, Any]

class PineconeRAG:
    """RAG system for answering questions about Ray Peat's work using Pinecone."""
    
    def __init__(self, search_engine: PineconeVectorSearch = None, index_name: str = "ray-peat-corpus"):
        """Initialize the Pinecone-based RAG system."""
        self.search_engine = search_engine or PineconeVectorSearch(index_name=index_name)
        self.llm_model = "gemini-2.5-flash"  # Fast and cost-effective for RAG
        
    async def answer_question(
        self, 
        question: str, 
        max_sources: int = 5,
        min_similarity: float = 0.3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """
        Answer a question using Pinecone-based RAG approach.
        
        Args:
            question: The user's question
            max_sources: Maximum number of source passages to use
            min_similarity: Minimum similarity threshold for sources
            metadata_filter: Optional metadata filters for Pinecone search
            
        Returns:
            RAGResponse with answer and sources
        """
        
        search_stats = {"method": "pinecone_vector_search"}
        
        try:
            # Step 1: Retrieve relevant passages from Pinecone
            search_results = await self.search_engine.search(
                query=question,
                top_k=max_sources,
                min_similarity=min_similarity,
                filter_dict=metadata_filter
            )
            
            search_stats.update({
                "results_found": len(search_results),
                "min_similarity": min_similarity,
                "metadata_filter": metadata_filter
            })
            
            if not search_results:
                return RAGResponse(
                    answer="I couldn't find any relevant information about that topic in Ray Peat's work.",
                    sources=[],
                    confidence=0.0,
                    query=question,
                    search_stats=search_stats
                )
            
            # Step 2: Generate answer using retrieved context
            answer = await self._generate_answer(question, search_results)
            
            # Step 3: Calculate confidence based on similarity scores
            avg_similarity = sum(result.similarity_score for result in search_results) / len(search_results)
            confidence = min(avg_similarity * 1.2, 1.0)  # Scale similarity to confidence
            
            search_stats.update({
                "avg_similarity": avg_similarity,
                "confidence": confidence,
                "sources_used": len(search_results)
            })
            
            return RAGResponse(
                answer=answer,
                sources=search_results,
                confidence=confidence,
                query=question,
                search_stats=search_stats
            )
            
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return RAGResponse(
                answer=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                confidence=0.0,
                query=question,
                search_stats={"error": str(e)}
            )
    
    async def answer_with_source_filter(
        self,
        question: str,
        source_files: List[str],
        max_sources: int = 5,
        min_similarity: float = 0.3
    ) -> RAGResponse:
        """
        Answer a question using only specific source files.
        
        Args:
            question: The user's question
            source_files: List of source file names to search within
            max_sources: Maximum number of source passages to use
            min_similarity: Minimum similarity threshold
            
        Returns:
            RAGResponse with answer and sources
        """
        # Create metadata filter for specific source files
        metadata_filter = {
            "source_file": {"$in": source_files}
        }
        
        return await self.answer_question(
            question=question,
            max_sources=max_sources,
            min_similarity=min_similarity,
            metadata_filter=metadata_filter
        )
    
    async def get_related_questions(
        self,
        topic: str,
        max_questions: int = 8,
        min_similarity: float = 0.2
    ) -> List[str]:
        """
        Get related questions/contexts based on a topic.
        
        Args:
            topic: The topic to find related questions for
            max_questions: Maximum number of questions to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of related question contexts
        """
        try:
            search_results = await self.search_engine.search(
                query=topic,
                top_k=max_questions * 2,
                min_similarity=min_similarity
            )
            
            # Extract unique contexts/questions
            questions = []
            seen_contexts = set()
            
            for result in search_results:
                context = result.context.strip()
                if context and context not in seen_contexts and len(context) > 10:
                    questions.append(context)
                    seen_contexts.add(context)
                    
                    if len(questions) >= max_questions:
                        break
            
            logger.info(f"Found {len(questions)} related questions for topic: {topic}")
            return questions
            
        except Exception as e:
            logger.error(f"Error getting related questions: {e}")
            return []
    
    async def find_similar_responses(
        self,
        response_text: str,
        max_similar: int = 5,
        min_similarity: float = 0.4
    ) -> List[SearchResult]:
        """
        Find Ray Peat responses similar to given text.
        
        Args:
            response_text: Text to find similar responses for
            max_similar: Maximum number of similar responses
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar SearchResult objects
        """
        try:
            # Use the response text as a query to find similar responses
            search_results = await self.search_engine.search(
                query=response_text,
                top_k=max_similar,
                min_similarity=min_similarity
            )
            
            logger.info(f"Found {len(search_results)} similar responses")
            return search_results
            
        except Exception as e:
            logger.error(f"Error finding similar responses: {e}")
            return []
    
    async def _generate_answer(self, question: str, sources: List[SearchResult]) -> str:
        """Generate an answer using the LLM with retrieved context."""
        
        # Build context from sources
        context_parts = []
        for i, source in enumerate(sources, 1):
            truncation_note = " [Note: Response was truncated due to length]" if source.truncated else ""
            
            context_parts.append(
                f"Source {i} (from {source.source_file}, similarity: {source.similarity_score:.3f}):\n"
                f"Context: {source.context}\n"
                f"Ray Peat's response: {source.ray_peat_response}{truncation_note}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # Create the prompt
        prompt = f"""You are an expert on Ray Peat's bioenergetic approach to health and biology. Answer the following question based ONLY on the provided sources from Ray Peat's work.

Question: {question}

Ray Peat's Knowledge Sources (from Pinecone vector database):
{context}

Instructions:
1. Answer based ONLY on the information provided in the sources above
2. If the sources don't contain enough information to answer the question, say so
3. Quote or reference specific parts of Ray Peat's responses when possible
4. Maintain Ray Peat's perspective and terminology (bioenergetic, metabolic rate, etc.)
5. Be accurate and don't make assumptions beyond what's stated
6. If multiple sources give different perspectives, acknowledge this
7. Consider the similarity scores - higher scores indicate more relevant sources
8. If a source is marked as truncated, note that the full response may contain additional information

Answer:"""

        try:
            answer = await self._call_gemini_llm(prompt)
            return answer or "I couldn't generate a response. Please try rephrasing your question."
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
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
                        logger.error(f"LLM API Error {response.status}: {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return None
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the underlying search engine."""
        return self.search_engine.get_corpus_stats()

# Global instance for the API (same pattern as the original)
rag_system = PineconeRAG()
