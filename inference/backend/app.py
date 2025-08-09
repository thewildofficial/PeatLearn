#!/usr/bin/env python3
"""
Ray Peat Legacy - Backend API Server

FastAPI application providing RAG-powered search and question answering.
"""

import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import settings
from rag import RayPeatVectorSearch, RayPeatRAG, SearchResult, RAGResponse

# Initialize RAG components
search_engine = RayPeatVectorSearch()
rag_system = RayPeatRAG(search_engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    query: str
    results: List[dict]
    total_results: int

class QuestionResponse(BaseModel):
    """Response model for question answering endpoint."""
    question: str
    answer: str
    confidence: float
    sources: List[dict]

class CorpusStatsResponse(BaseModel):
    """Response model for corpus statistics."""
    total_embeddings: int
    total_tokens: int
    embedding_dimensions: int
    source_files: int
    files_breakdown: dict

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "status": "healthy",
        "version": settings.VERSION,
        "corpus_loaded": search_engine.embeddings is not None
    }

@app.get("/api/search", response_model=SearchResponse)
async def search_corpus(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Number of results to return"),
    min_similarity: float = Query(0.1, ge=0.0, le=1.0, description="Minimum similarity threshold")
):
    """Search the Ray Peat corpus using semantic search."""
    try:
        results = await search_engine.search(
            query=q,
            top_k=limit,
            min_similarity=min_similarity
        )
        
        # Convert results to dict format
        results_dict = [
            {
                "id": result.id,
                "context": result.context,
                "ray_peat_response": result.ray_peat_response,
                "source_file": result.source_file,
                "similarity_score": result.similarity_score,
                "tokens": result.tokens
            }
            for result in results
        ]
        
        return SearchResponse(
            query=q,
            results=results_dict,
            total_results=len(results_dict)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/api/ask", response_model=QuestionResponse)
async def ask_question(
    question: str = Query(..., description="Question to ask Ray Peat's knowledge base"),
    max_sources: int = Query(5, ge=1, le=10, description="Maximum number of sources to use"),
    min_similarity: float = Query(0.3, ge=0.0, le=1.0, description="Minimum similarity for sources")
):
    """Ask a question and get an AI-generated answer based on Ray Peat's work."""
    try:
        rag_response = await rag_system.answer_question(
            question=question,
            max_sources=max_sources,
            min_similarity=min_similarity
        )
        
        # Convert sources to dict format
        sources_dict = [
            {
                "id": source.id,
                "context": source.context,
                "ray_peat_response": source.ray_peat_response,
                "source_file": source.source_file,
                "similarity_score": source.similarity_score,
                "tokens": source.tokens
            }
            for source in rag_response.sources
        ]
        
        return QuestionResponse(
            question=question,
            answer=rag_response.answer,
            confidence=rag_response.confidence,
            sources=sources_dict
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question answering error: {str(e)}")

@app.get("/api/stats", response_model=CorpusStatsResponse)
async def get_corpus_stats():
    """Get statistics about the loaded Ray Peat corpus."""
    try:
        stats = search_engine.get_corpus_stats()
        
        if "error" in stats:
            raise HTTPException(status_code=503, detail=stats["error"])
        
        return CorpusStatsResponse(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

@app.get("/api/related")
async def get_related_topics(
    query: str = Query(..., description="Query to find related topics for"),
    limit: int = Query(8, ge=1, le=20, description="Number of related topics to return")
):
    """Get topics related to the query."""
    try:
        topics = await rag_system.get_related_topics(query, max_topics=limit)
        return {
            "query": query,
            "related_topics": topics,
            "total_topics": len(topics)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Related topics error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/api/topics")
async def get_topics():
    """Get available topics and categories."""
    return {
        "topics": [
            "Thyroid Function",
            "Nutrition",
            "Hormones",
            "Metabolism",
            "Stress",
            "Supplements"
        ],
        "message": "Topics will be dynamically generated from corpus analysis"
    }