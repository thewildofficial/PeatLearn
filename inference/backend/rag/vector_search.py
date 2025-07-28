#!/usr/bin/env python3
"""
Vector Search Engine for Ray Peat Corpus

Provides semantic search capabilities using the generated embeddings.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import aiohttp
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings

@dataclass
class SearchResult:
    """Represents a search result from the corpus."""
    id: str
    context: str
    ray_peat_response: str
    source_file: str
    similarity_score: float
    tokens: int

class RayPeatVectorSearch:
    """Vector search engine for the Ray Peat corpus."""
    
    def __init__(self, vectors_dir: Path = None):
        """Initialize the vector search engine."""
        if vectors_dir is None:
            vectors_dir = Path(__file__).parent.parent.parent.parent / "embedding" / "vectors"
        
        self.vectors_dir = vectors_dir
        self.embeddings = None
        self.metadata = None
        self.embedding_model = "gemini-embedding-001"
        self.embedding_dimensions = 768
        
        # Load existing embeddings on initialization
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load the pre-computed embeddings and metadata."""
        try:
            # Find the latest embedding files
            embedding_files = list(self.vectors_dir.glob("embeddings_*.npy"))
            metadata_files = list(self.vectors_dir.glob("metadata_*.json"))
            
            if not embedding_files or not metadata_files:
                raise FileNotFoundError("No embedding files found")
            
            # Get the latest files (assume they have timestamps)
            latest_embedding = max(embedding_files, key=lambda x: x.stat().st_mtime)
            latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
            
            # Load embeddings
            self.embeddings = np.load(latest_embedding)
            
            # Load metadata
            with open(latest_metadata, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            print(f"Loaded {len(self.embeddings)} embeddings from {latest_embedding.name}")
            print(f"Loaded {len(self.metadata)} metadata entries from {latest_metadata.name}")
            
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            self.embeddings = None
            self.metadata = None
    
    async def generate_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Generate embedding for a search query using Gemini API."""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.embedding_model}:embedContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": settings.GEMINI_API_KEY
        }
        
        payload = {
            "model": f"models/{self.embedding_model}",
            "content": {
                "parts": [{"text": query}]
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        embedding = result["embedding"]["values"]
                        return np.array(embedding)
                    else:
                        error_text = await response.text()
                        print(f"API Error {response.status}: {error_text}")
                        return None
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return None
    
    async def search(self, query: str, top_k: int = 10, min_similarity: float = 0.1) -> List[SearchResult]:
        """
        Search the corpus for passages most similar to the query.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of SearchResult objects
        """
        if self.embeddings is None or self.metadata is None:
            raise ValueError("Embeddings not loaded. Please ensure embedding files exist.")
        
        # Generate embedding for the query
        query_embedding = await self.generate_query_embedding(query)
        if query_embedding is None:
            return []
        
        # Calculate cosine similarities
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k results above minimum similarity
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more than needed for filtering
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity < min_similarity:
                continue
            
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                result = SearchResult(
                    id=meta["id"],
                    context=meta["context"],
                    ray_peat_response=meta["ray_peat_response"],
                    source_file=meta["source_file"],
                    similarity_score=float(similarity),
                    tokens=meta["tokens"]
                )
                results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded corpus."""
        if self.embeddings is None or self.metadata is None:
            return {"error": "No embeddings loaded"}
        
        source_files = {}
        total_tokens = 0
        
        for meta in self.metadata:
            source = meta["source_file"]
            source_files[source] = source_files.get(source, 0) + 1
            total_tokens += meta["tokens"]
        
        return {
            "total_embeddings": len(self.embeddings),
            "total_tokens": total_tokens,
            "embedding_dimensions": self.embeddings.shape[1] if len(self.embeddings) > 0 else 0,
            "source_files": len(source_files),
            "files_breakdown": source_files
        }

# Global instance for the API
search_engine = RayPeatVectorSearch()
