#!/usr/bin/env python3
"""
Ray Peat Legacy - Backend API Server

FastAPI application providing RAG-powered search and question answering.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings

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

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "status": "healthy",
        "version": settings.VERSION
    }

@app.get("/api/search")
async def search_corpus(q: str):
    """Search the Ray Peat corpus."""
    return {
        "query": q,
        "results": [],
        "message": "Search functionality will be implemented with RAG system"
    }

@app.post("/api/ask")
async def ask_question(question: str, context: str = None):
    """Ask a question about Ray Peat's work."""
    return {
        "question": question,
        "answer": "Question answering will be implemented with fine-tuned LLM",
        "context": context,
        "sources": []
    }

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.API_HOST, 
        port=settings.API_PORT,
        reload=settings.DEBUG
    ) 