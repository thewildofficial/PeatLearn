#!/usr/bin/env python3
"""
Test script for the Ray Peat RAG system.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from inference.backend.rag import RayPeatVectorSearch, RayPeatRAG

async def test_vector_search():
    """Test the vector search functionality."""
    print("🔍 Testing Vector Search...")
    
    search_engine = RayPeatVectorSearch()
    
    # Test corpus loading
    print(f"📊 Corpus stats: {search_engine.get_corpus_stats()}")
    
    # Test search
    query = "thyroid hormone metabolism"
    print(f"\n🔎 Searching for: '{query}'")
    
    results = await search_engine.search(query, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (similarity: {result.similarity_score:.3f}) ---")
        print(f"Context: {result.context[:100]}...")
        print(f"Ray Peat: {result.ray_peat_response[:150]}...")
        print(f"Source: {result.source_file}")

async def test_rag_system():
    """Test the RAG question answering."""
    print("\n🤖 Testing RAG System...")
    
    rag = RayPeatRAG()
    
    question = "What does Ray Peat say about thyroid function?"
    print(f"\n❓ Question: {question}")
    
    response = await rag.answer_question(question)
    
    print(f"\n✅ Answer (confidence: {response.confidence:.2f}):")
    print(response.answer)
    
    print(f"\n📚 Based on {len(response.sources)} sources:")
    for i, source in enumerate(response.sources, 1):
        print(f"  {i}. {source.source_file} (similarity: {source.similarity_score:.3f})")

async def main():
    """Run all tests."""
    print("🧪 Testing Ray Peat RAG System\n" + "="*50)
    
    try:
        await test_vector_search()
        await test_rag_system()
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
