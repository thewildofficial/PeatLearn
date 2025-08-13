#!/usr/bin/env python3
"""
DEPRECATED: Legacy RAG (file-based embeddings)

This module has been superseded by the Pinecone-backed RAG implementation in
`embedding/pinecone/rag_system.py`. Please migrate imports to the new module.
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
        """Call Gemini LLM with continuation to reduce truncation."""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        async def call_once(user_prompt: str) -> tuple[str, str]:
            try:
                import google.generativeai as genai  # type: ignore
                genai.configure(api_key=settings.GEMINI_API_KEY)
                model_name = self.llm_model if self.llm_model.startswith("gemini") else f"models/{self.llm_model}"
                model = genai.GenerativeModel(model_name)
                resp = await asyncio.to_thread(
                    model.generate_content,
                    user_prompt,
                    generation_config={
                        "temperature": 0.3,
                        "max_output_tokens": 4096,
                        "top_p": 0.8,
                        "top_k": 40,
                    },
                )
                text = getattr(resp, "text", "") or ""
                finish = ""
                try:
                    finish = str(resp.candidates[0].finish_reason)  # type: ignore[attr-defined]
                except Exception:
                    finish = ""
                return text, finish
            except Exception:
                pass

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.llm_model}:generateContent"
            headers = {"Content-Type": "application/json", "x-goog-api-key": settings.GEMINI_API_KEY}
            payload = {
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096, "topP": 0.8, "topK": 40},
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            text = ""
                            finish = ""
                            try:
                                candidates = result.get("candidates", [])
                                if candidates:
                                    first = candidates[0]
                                    finish = str(first.get("finishReason", ""))
                                    content = first.get("content", {})
                                    parts = content.get("parts", []) if isinstance(content, dict) else []
                                    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
                                    text = "".join(texts)
                            except Exception:
                                pass
                            if not text and isinstance(result, dict):
                                text = result.get("text") or result.get("output_text") or ""
                            return text, finish
                        else:
                            return "", ""
            except Exception:
                return "", ""

        accumulated = ""
        loops = 0
        max_chars = 12000
        first, finish = await call_once(prompt)
        accumulated += first or ""
        while loops < 4 and len(accumulated) < max_chars:
            seems_cut = not accumulated.strip().endswith(('.', '"', "'", '}', ']', ')'))
            if finish and finish.upper() != 'MAX_TOKENS' and not seems_cut:
                break
            loops += 1
            tail = accumulated[-600:]
            cont = f"Continue the previous answer. Continue seamlessly without repeating.\nContext tail: {tail}"
            more, finish = await call_once(cont)
            if not more:
                break
            accumulated += ("\n" if not accumulated.endswith("\n") else "") + more
        return accumulated or None
    
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
