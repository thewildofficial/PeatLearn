#!/usr/bin/env python3
"""
Test Script for Pinecone Vector Search and RAG System

Comprehensive testing of the Pinecone migration functionality.
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

# Resolve project root and ensure it's on sys.path for direct execution
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Load .env if present so PINECONE_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except Exception:
    pass

# Robust import block: support both module and script execution
try:
    from .vector_search import PineconeVectorSearch
    from .rag_system import PineconeRAG
    from .utils import PineconeManager
except Exception:
    try:
        from embedding.pinecone.vector_search import PineconeVectorSearch
        from embedding.pinecone.rag_system import PineconeRAG
        from embedding.pinecone.utils import PineconeManager
    except Exception as e:
        print(f"Import error. Ensure you're running with the project root on PYTHONPATH or use 'python -m embedding.pinecone.test_pinecone'. Underlying error: {e}")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

class PineconeTestSuite:
    """Test suite for Pinecone functionality."""
    
    def __init__(self, index_name: str = "ray-peat-corpus"):
        self.index_name = index_name
        self.search_engine = None
        self.rag_system = None
        self.manager = None
        
        self.test_results = {
            "setup": False,
            "vector_search": False,
            "rag_system": False,
            "utilities": False,
            "performance": {},
            "errors": []
        }
    
    async def setup_test_environment(self) -> bool:
        """Initialize all components for testing."""
        try:
            print("🔧 Setting up test environment...")
            
            # Initialize components
            self.search_engine = PineconeVectorSearch(self.index_name)
            self.rag_system = PineconeRAG(self.search_engine, self.index_name)
            self.manager = PineconeManager(self.index_name)
            
            # Check if index exists and has data
            stats = self.manager.get_index_info()
            if "error" in stats:
                raise Exception(f"Index setup error: {stats['error']}")
            
            vector_count = stats.get("total_vector_count", 0)
            if vector_count == 0:
                raise Exception("Index has no vectors. Please run upload.py first.")
            
            print(f"✅ Connected to index '{self.index_name}' with {vector_count:,} vectors")
            self.test_results["setup"] = True
            return True
            
        except Exception as e:
            error_msg = f"Setup failed: {e}"
            print(f"❌ {error_msg}")
            self.test_results["errors"].append(error_msg)
            return False
    
    async def test_vector_search(self) -> bool:
        """Test vector search functionality."""
        try:
            print("\n🔍 Testing vector search functionality...")
            
            test_queries = [
                "thyroid hormone metabolism",
                "stress and cortisol effects",
                "sugar and energy production",
                "estrogen dominance problems",
                "mitochondrial function"
            ]
            
            search_results = {}
            
            for query in test_queries:
                print(f"  Testing query: '{query}'")
                
                results = await self.search_engine.search(
                    query=query,
                    top_k=5,
                    min_similarity=0.1
                )
                
                search_results[query] = {
                    "result_count": len(results),
                    "avg_similarity": sum(r.similarity_score for r in results) / len(results) if results else 0,
                    "top_similarity": max(r.similarity_score for r in results) if results else 0
                }
                
                print(f"    Found {len(results)} results, avg similarity: {search_results[query]['avg_similarity']:.3f}")
                
                # Verify result structure
                if results:
                    first_result = results[0]
                    assert hasattr(first_result, 'context'), "Result missing context"
                    assert hasattr(first_result, 'ray_peat_response'), "Result missing response"
                    assert hasattr(first_result, 'similarity_score'), "Result missing similarity score"
                    assert hasattr(first_result, 'source_file'), "Result missing source file"
            
            # Test metadata filtering
            print("  Testing metadata filtering...")
            filter_results = await self.search_engine.search(
                query="health advice",
                top_k=10,
                filter_dict={"tokens": {"$gte": 100}}  # Find longer responses
            )
            
            print(f"    Metadata filter found {len(filter_results)} results")
            
            self.test_results["vector_search"] = True
            self.test_results["performance"]["search_results"] = search_results
            print("✅ Vector search tests passed")
            return True
            
        except Exception as e:
            error_msg = f"Vector search test failed: {e}"
            print(f"❌ {error_msg}")
            self.test_results["errors"].append(error_msg)
            return False
    
    async def test_rag_system(self) -> bool:
        """Test RAG system functionality."""
        try:
            print("\n🤖 Testing RAG system functionality...")
            
            test_questions = [
                "What did Ray Peat say about thyroid function?",
                "How does sugar affect metabolism according to Ray Peat?",
                "What are Ray Peat's views on stress hormones?",
                "What did Ray Peat recommend for improving energy?"
            ]
            
            rag_results = {}
            
            for question in test_questions:
                print(f"  Testing question: '{question[:50]}...'")
                
                response = await self.rag_system.answer_question(
                    question=question,
                    max_sources=3,
                    min_similarity=0.2
                )
                
                rag_results[question] = {
                    "has_answer": bool(response.answer and len(response.answer) > 10),
                    "source_count": len(response.sources),
                    "confidence": response.confidence,
                    "answer_length": len(response.answer) if response.answer else 0
                }
                
                print(f"    Answer length: {rag_results[question]['answer_length']} chars, "
                      f"confidence: {rag_results[question]['confidence']:.3f}, "
                      f"sources: {rag_results[question]['source_count']}")
                
                # Verify response structure
                assert hasattr(response, 'answer'), "Response missing answer"
                assert hasattr(response, 'sources'), "Response missing sources"
                assert hasattr(response, 'confidence'), "Response missing confidence"
                assert hasattr(response, 'search_stats'), "Response missing search stats"
            
            # Test related questions
            print("  Testing related questions feature...")
            related = await self.rag_system.get_related_questions("metabolism", max_questions=5)
            print(f"    Found {len(related)} related questions")
            
            # Test source filtering
            print("  Testing source file filtering...")
            source_files = self.manager.get_unique_source_files()[:3]  # Test with first 3 source files
            if source_files:
                filtered_response = await self.rag_system.answer_with_source_filter(
                    question="What is important for health?",
                    source_files=source_files,
                    max_sources=3
                )
                print(f"    Source-filtered response: {len(filtered_response.answer)} chars")
            
            self.test_results["rag_system"] = True
            self.test_results["performance"]["rag_results"] = rag_results
            print("✅ RAG system tests passed")
            return True
            
        except Exception as e:
            error_msg = f"RAG system test failed: {e}"
            print(f"❌ {error_msg}")
            self.test_results["errors"].append(error_msg)
            return False
    
    async def test_utilities(self) -> bool:
        """Test utility functions."""
        try:
            print("\n🛠️ Testing utility functions...")
            
            # Test index info
            info = self.manager.get_index_info()
            assert "total_vector_count" in info, "Index info missing vector count"
            print(f"  Index info: {info['total_vector_count']} vectors")
            
            # Test sampling
            samples = self.manager.sample_vectors(3)
            assert len(samples) > 0, "Sampling returned no results"
            print(f"  Sampled {len(samples)} vectors")
            
            # Test integrity verification
            verification = self.manager.verify_vector_integrity(10)
            assert "total_sampled" in verification, "Verification missing sample count"
            print(f"  Verified {verification['total_sampled']} vectors")
            
            # Test source file listing
            source_files = self.manager.get_unique_source_files()
            assert len(source_files) > 0, "No source files found"
            print(f"  Found {len(source_files)} unique source files")
            
            # Test health report
            health_report = self.manager.generate_health_report()
            assert "health_score" in health_report or "error" in health_report["integrity_check"], "Health report incomplete"
            print(f"  Generated health report")
            
            self.test_results["utilities"] = True
            self.test_results["performance"]["utility_stats"] = {
                "vector_count": info.get("total_vector_count", 0),
                "source_files": len(source_files),
                "sample_size": len(samples)
            }
            print("✅ Utility tests passed")
            return True
            
        except Exception as e:
            error_msg = f"Utility test failed: {e}"
            print(f"❌ {error_msg}")
            self.test_results["errors"].append(error_msg)
            return False
    
    async def run_performance_benchmarks(self) -> bool:
        """Run performance benchmarks."""
        try:
            print("\n⚡ Running performance benchmarks...")
            
            import time
            
            # Benchmark search performance
            start_time = time.time()
            search_results = await self.search_engine.search("energy metabolism", top_k=10)
            search_time = time.time() - start_time
            
            print(f"  Search latency: {search_time:.3f}s for {len(search_results)} results")
            
            # Benchmark RAG performance
            start_time = time.time()
            rag_response = await self.rag_system.answer_question(
                "What did Ray Peat say about energy?",
                max_sources=5
            )
            rag_time = time.time() - start_time
            
            print(f"  RAG latency: {rag_time:.3f}s for answer ({len(rag_response.answer)} chars)")
            
            self.test_results["performance"]["benchmarks"] = {
                "search_latency_seconds": search_time,
                "rag_latency_seconds": rag_time,
                "search_results_count": len(search_results),
                "rag_answer_length": len(rag_response.answer)
            }
            
            print("✅ Performance benchmarks completed")
            return True
            
        except Exception as e:
            error_msg = f"Performance benchmark failed: {e}"
            print(f"❌ {error_msg}")
            self.test_results["errors"].append(error_msg)
            return False
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report."""
        report = {
            "test_time": str(Path(__file__).parent.parent.parent),  # Placeholder for timestamp
            "index_name": self.index_name,
            "test_results": self.test_results,
            "summary": {
                "tests_passed": sum([
                    self.test_results["setup"],
                    self.test_results["vector_search"],
                    self.test_results["rag_system"],
                    self.test_results["utilities"]
                ]),
                "total_tests": 4,
                "success_rate": 0,
                "has_errors": len(self.test_results["errors"]) > 0
            }
        }
        
        report["summary"]["success_rate"] = report["summary"]["tests_passed"] / report["summary"]["total_tests"]
        
        return report
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results."""
        print("🚀 Starting Pinecone Test Suite")
        print("=" * 50)
        
        # Run tests in sequence
        if await self.setup_test_environment():
            await self.test_vector_search()
            await self.test_rag_system()
            await self.test_utilities()
            await self.run_performance_benchmarks()
        
        # Generate report
        report = self.generate_test_report()
        
        print(f"\n📊 Test Summary:")
        print(f"   Tests passed: {report['summary']['tests_passed']}/{report['summary']['total_tests']}")
        print(f"   Success rate: {report['summary']['success_rate']:.1%}")
        
        if report['summary']['has_errors']:
            print(f"   Errors: {len(self.test_results['errors'])}")
            for error in self.test_results['errors']:
                print(f"     - {error}")
        
        return report

async def main():
    """Main test function."""
    try:
        test_suite = PineconeTestSuite()
        report = await test_suite.run_all_tests()
        
        # Save test report
        report_file = Path(__file__).parent / f"test_report_{report['index_name']}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Test report saved to: {report_file}")
        
        # Exit with appropriate code
        if report['summary']['success_rate'] == 1.0:
            print("🎉 All tests passed!")
            return 0
        else:
            print("⚠️ Some tests failed.")
            return 1
            
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

