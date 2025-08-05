#!/usr/bin/env python3
"""
Integration test for the RayPeatRAG system.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "inference" / "backend"))

# Set environment variable for settings
os.environ['PROJECT_ROOT'] = str(project_root)

from inference.backend.rag.rag_system import RayPeatRAG
from config.settings import settings

async def test_rag_answer_question():
    """Test if the RAG system can answer a question and provide sources."""
    print("\n🧪 Running RAG System Integration Test...")
    
    if not settings.GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY is not set. Please set it as an environment variable to run this test.")
        return False

    try:
        rag = RayPeatRAG()
        print("✅ RayPeatRAG initialized.")

        question = "What does Ray Peat say about the benefits of progesterone?"
        print(f"❓ Question: '{question}'")

        response = await rag.answer_question(question, max_sources=3)

        print(f"✅ Answer generated. Confidence: {response.confidence:.2f}")
        print(f"📝 Answer: {response.answer[:200]}...") # Print first 200 chars of answer
        print(f"📚 Sources found: {len(response.sources)}")

        assert response.answer is not None and len(response.answer) > 0, "RAG system returned an empty answer."
        assert len(response.sources) > 0, "RAG system found no sources."
        
        print("\n--- Sources ---")
        for i, source in enumerate(response.sources):
            print(f"Source {i+1}: {source.source_file} (Similarity: {source.similarity_score:.3f})")
            print(f"Context: {source.context[:100]}...") # Print first 100 chars of context

        print("\n🎉 RAG System Integration Test Passed!")
        return True

    except Exception as e:
        print(f"❌ RAG System Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    exit_code = 0
    if not await test_rag_answer_question():
        exit_code = 1
    sys.exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main())
