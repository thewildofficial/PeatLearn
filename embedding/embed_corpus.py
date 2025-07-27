#!/usr/bin/env python3
"""
Ray Peat Legacy - Gemini Text Embedding Pipeline

This module generates embeddings for the cleaned Ray Peat corpus using Google's Gemini API.
Processes Q&A pairs and creates vector representations for semantic search.
"""

import sys
import json
import time
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pickle
from datetime import datetime
import logging
import re
import hashlib

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings, PATHS

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QAPair:
    """Represents a single Q&A pair from the corpus."""
    id: str
    context: str
    ray_peat_response: str
    source_file: str
    tokens: int
    embedding: Optional[np.ndarray] = None

@dataclass
class EmbeddingProgress:
    """Tracks embedding generation progress."""
    total_pairs: int = 0
    processed_pairs: int = 0
    successful_embeddings: int = 0
    failed_embeddings: int = 0
    total_cost: float = 0.0
    start_time: datetime = None
    
    def progress_percentage(self) -> float:
        return (self.processed_pairs / self.total_pairs * 100) if self.total_pairs > 0 else 0
    
    def estimated_time_remaining(self) -> float:
        if self.processed_pairs == 0:
            return 0
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.processed_pairs / elapsed
        remaining_pairs = self.total_pairs - self.processed_pairs
        return remaining_pairs / rate if rate > 0 else 0

class GeminiEmbeddingGenerator:
    """Handles embedding generation using Google's Gemini API."""
    
    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.rate_limit_delay = 60 / 90  # 90 RPM for safety (free tier: 100 RPM)
        self.last_request_time = 0
        
        # Gemini embedding pricing (per 1M tokens)
        self.cost_per_million_tokens = 0.15  # $0.15 per 1M tokens (official pricing)
        
    async def _rate_limit_wait(self):
        """Ensure we don't exceed rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            await asyncio.sleep(sleep_time)
        self.last_request_time = time.time()
    
    async def generate_embedding(self, text: str, session: aiohttp.ClientSession) -> Optional[np.ndarray]:
        """Generate embedding for a single text."""
        await self._rate_limit_wait()
        
        url = f"{self.base_url}/models/{self.model}:embedContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        payload = {
            "model": f"models/{self.model}",
            "content": {
                "parts": [{"text": text}]
            }
        }
        
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    embedding = data.get("embedding", {}).get("values", [])
                    return np.array(embedding, dtype=np.float32)
                else:
                    error_text = await response.text()
                    logger.error(f"API Error {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return None
    
    def calculate_cost(self, total_tokens: int) -> float:
        """Calculate embedding cost based on tokens."""
        return (total_tokens / 1_000_000) * self.cost_per_million_tokens

class RayPeatCorpusEmbedder:
    """Main class for embedding the Ray Peat corpus."""
    
    def __init__(self):
        self.processed_data_dir = PATHS["processed_data"] / "ai_cleaned"
        self.embedding_output_dir = PATHS["embedding"] / "vectors"
        self.embedding_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint files for resume functionality
        self.checkpoint_file = self.embedding_output_dir / "checkpoint.json"
        self.completed_pairs_file = self.embedding_output_dir / "completed_pairs.pkl"
        
        # Initialize Gemini embedder
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.embedder = GeminiEmbeddingGenerator(
            api_key=settings.GEMINI_API_KEY,
            model=settings.EMBEDDING_MODEL
        )
        
        self.progress = EmbeddingProgress()
        
    def parse_qa_pairs_from_file(self, file_path: Path) -> List[QAPair]:
        """Parse Q&A pairs from a single processed file."""
        pairs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by double newlines to get individual Q&A pairs
            sections = content.split('\n\n')
            
            pair_id = 0
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                
                # Look for Ray Peat response pattern
                ray_peat_match = re.search(r'\*\*RAY PEAT:\*\*\s*(.*?)(?=\*\*CONTEXT:\*\*|$)', section, re.DOTALL)
                context_match = re.search(r'\*\*CONTEXT:\*\*\s*(.*?)(?=\*\*RAY PEAT:\*\*|$)', section, re.DOTALL)
                
                if ray_peat_match:
                    ray_peat_response = ray_peat_match.group(1).strip()
                    context = context_match.group(1).strip() if context_match else ""
                    
                    # Create combined text for embedding
                    combined_text = f"Context: {context}\n\nRay Peat: {ray_peat_response}"
                    
                    # Estimate tokens (rough approximation)
                    tokens = len(combined_text.split())
                    
                    pair = QAPair(
                        id=f"{file_path.stem}_{pair_id}",
                        context=context,
                        ray_peat_response=ray_peat_response,
                        source_file=str(file_path.relative_to(self.processed_data_dir)),
                        tokens=tokens
                    )
                    pairs.append(pair)
                    pair_id += 1
                    
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            
        return pairs
    
    def load_all_qa_pairs(self) -> List[QAPair]:
        """Load all Q&A pairs from the processed corpus."""
        logger.info("Loading Q&A pairs from processed corpus...")
        
        all_pairs = []
        processed_files = list(self.processed_data_dir.rglob("*_processed.txt"))
        
        logger.info(f"Found {len(processed_files)} processed files")
        
        for file_path in processed_files:
            pairs = self.parse_qa_pairs_from_file(file_path)
            all_pairs.extend(pairs)
            logger.debug(f"Loaded {len(pairs)} pairs from {file_path.name}")
        
        logger.info(f"Total Q&A pairs loaded: {len(all_pairs)}")
        return all_pairs
    
    async def generate_embeddings_batch(self, qa_pairs: List[QAPair], batch_size: int = 10, checkpoint_interval: int = 100) -> List[QAPair]:
        """Generate embeddings for a batch of Q&A pairs with checkpoint support."""
        
        # Load existing checkpoint
        completed_pairs, checkpoint_data = self.load_checkpoint()
        completed_hashes = set(checkpoint_data.get("completed_hashes", []))
        
        # Filter out already completed pairs
        remaining_pairs = [pair for pair in qa_pairs if not self.is_pair_completed(pair, completed_hashes)]
        
        total_pairs = len(qa_pairs)
        already_completed = len(completed_pairs)
        
        if already_completed > 0:
            logger.info(f"Resuming from checkpoint: {already_completed}/{total_pairs} pairs already completed")
            logger.info(f"Remaining to process: {len(remaining_pairs)} pairs")
            self.progress.total_cost = checkpoint_data.get("total_cost", 0.0)
        else:
            logger.info(f"Starting fresh: {len(remaining_pairs)} Q&A pairs to process")
        
        if not remaining_pairs:
            logger.info("All pairs already completed!")
            return completed_pairs
        
        self.progress.total_pairs = total_pairs
        self.progress.processed_pairs = already_completed
        self.progress.successful_embeddings = already_completed
        self.progress.start_time = datetime.now()
        
        # Combine completed pairs with new ones
        all_completed_pairs = completed_pairs.copy()
        
        async with aiohttp.ClientSession() as session:
            for i, pair in enumerate(remaining_pairs):
                # Combine context and response for embedding
                text_to_embed = f"Context: {pair.context}\n\nRay Peat: {pair.ray_peat_response}"
                
                embedding = await self.embedder.generate_embedding(text_to_embed, session)
                
                if embedding is not None:
                    pair.embedding = embedding
                    all_completed_pairs.append(pair)
                    self.progress.successful_embeddings += 1
                else:
                    self.progress.failed_embeddings += 1
                    logger.warning(f"Failed to generate embedding for pair {pair.id}")
                
                self.progress.processed_pairs += 1
                
                # Calculate cost
                self.progress.total_cost += self.embedder.calculate_cost(pair.tokens)
                
                # Save checkpoint periodically
                if (i + 1) % checkpoint_interval == 0 or (i + 1) == len(remaining_pairs):
                    self.save_checkpoint(all_completed_pairs, total_pairs)
                
                # Progress update
                if (i + 1) % (checkpoint_interval // 10) == 0 or (i + 1) == len(remaining_pairs):
                    progress_pct = self.progress.progress_percentage()
                    eta_minutes = self.progress.estimated_time_remaining() / 60
                    logger.info(
                        f"Progress: {progress_pct:.1f}% "
                        f"({self.progress.processed_pairs}/{self.progress.total_pairs}) "
                        f"- ETA: {eta_minutes:.1f}m "
                        f"- Cost: ${self.progress.total_cost:.4f}"
                    )
        
        return all_completed_pairs
    
    def generate_pair_hash(self, pair: QAPair) -> str:
        """Generate a unique hash for a Q&A pair."""
        content = f"{pair.context}|{pair.ray_peat_response}|{pair.source_file}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def save_checkpoint(self, completed_pairs: List[QAPair], total_pairs: int):
        """Save progress checkpoint."""
        checkpoint_data = {
            "completed_count": len(completed_pairs),
            "total_count": total_pairs,
            "completed_hashes": [self.generate_pair_hash(pair) for pair in completed_pairs],
            "last_updated": datetime.now().isoformat(),
            "model": self.embedder.model,
            "total_cost": self.progress.total_cost
        }
        
        # Save checkpoint metadata
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Save completed pairs with embeddings
        with open(self.completed_pairs_file, 'wb') as f:
            pickle.dump(completed_pairs, f)
        
        logger.info(f"Checkpoint saved: {len(completed_pairs)}/{total_pairs} pairs completed")
    
    def load_checkpoint(self) -> Tuple[List[QAPair], Dict]:
        """Load previous progress if available."""
        if not self.checkpoint_file.exists():
            return [], {}
        
        try:
            # Load checkpoint metadata
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Load completed pairs
            if self.completed_pairs_file.exists():
                with open(self.completed_pairs_file, 'rb') as f:
                    completed_pairs = pickle.load(f)
                
                logger.info(f"Loaded checkpoint: {len(completed_pairs)} pairs already completed")
                return completed_pairs, checkpoint_data
            
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
        
        return [], {}
    
    def is_pair_completed(self, pair: QAPair, completed_hashes: set) -> bool:
        """Check if a pair has already been completed."""
        return self.generate_pair_hash(pair) in completed_hashes

    def save_embeddings(self, qa_pairs: List[QAPair], format: str = "pickle"):
        """Save embeddings to disk in specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "pickle":
            # Save complete QA pairs with embeddings
            output_file = self.embedding_output_dir / f"ray_peat_embeddings_{timestamp}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(qa_pairs, f)
            logger.info(f"Saved embeddings to {output_file}")
            
        elif format == "numpy":
            # Save embeddings as numpy arrays with metadata
            embeddings = np.array([pair.embedding for pair in qa_pairs if pair.embedding is not None])
            metadata = [
                {
                    "id": pair.id,
                    "context": pair.context,
                    "ray_peat_response": pair.ray_peat_response,
                    "source_file": pair.source_file,
                    "tokens": pair.tokens
                }
                for pair in qa_pairs if pair.embedding is not None
            ]
            
            np.save(self.embedding_output_dir / f"embeddings_{timestamp}.npy", embeddings)
            with open(self.embedding_output_dir / f"metadata_{timestamp}.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved {len(embeddings)} embeddings and metadata")
    
    def save_progress_report(self):
        """Save a detailed progress report."""
        report = {
            "generation_completed": datetime.now().isoformat(),
            "total_pairs": self.progress.total_pairs,
            "successful_embeddings": self.progress.successful_embeddings,
            "failed_embeddings": self.progress.failed_embeddings,
            "success_rate": self.progress.successful_embeddings / self.progress.total_pairs * 100,
            "total_cost_usd": round(self.progress.total_cost, 4),
            "embedding_model": self.embedder.model,
            "embedding_dimensions": settings.EMBEDDING_DIMENSIONS,
            "processing_time_minutes": (datetime.now() - self.progress.start_time).total_seconds() / 60
        }
        
        report_file = self.embedding_output_dir / "embedding_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Progress report saved to {report_file}")
        return report

async def main():
    """Main embedding pipeline function with automatic checkpoint detection and resume."""
    print("🚀 Ray Peat Legacy - Gemini Embedding Pipeline")
    print("=" * 60)
    
    try:
        # Initialize embedder
        embedder = RayPeatCorpusEmbedder()
        
        # Check for existing checkpoint first
        completed_pairs, checkpoint_data = embedder.load_checkpoint()
        
        # Load all Q&A pairs
        qa_pairs = embedder.load_all_qa_pairs()
        
        if not qa_pairs:
            logger.error("No Q&A pairs found. Check your processed data directory.")
            return
        
        # Calculate statistics
        total_tokens = sum(pair.tokens for pair in qa_pairs)
        estimated_total_cost = embedder.embedder.calculate_cost(total_tokens)
        
        # Show checkpoint status if exists
        if completed_pairs:
            already_completed = len(completed_pairs)
            progress_pct = (already_completed / len(qa_pairs) * 100)
            remaining_pairs = len(qa_pairs) - already_completed
            cost_so_far = checkpoint_data.get("total_cost", 0.0)
            last_updated = checkpoint_data.get("last_updated", "Unknown")
            
            print(f"🔄 Checkpoint Found - Resuming Progress:")
            print(f"   • Progress: {already_completed:,}/{len(qa_pairs):,} pairs ({progress_pct:.1f}%)")
            print(f"   • Remaining: {remaining_pairs:,} pairs")
            print(f"   • Cost so far: ${cost_so_far:.4f}")
            print(f"   • Last updated: {last_updated}")
            print(f"   • Embedding model: {embedder.embedder.model}")
            print(f"   • Rate limit: 90 RPM")
            print()
            
            if remaining_pairs == 0:
                print("✅ All embeddings already completed!")
                return
            else:
                print("🚀 Automatically resuming embedding generation...")
                
        else:
            print(f"📊 Starting Fresh - Corpus Statistics:")
            print(f"   • Total Q&A pairs: {len(qa_pairs):,}")
            print(f"   • Total tokens: {total_tokens:,}")
            print(f"   • Estimated total cost: ${estimated_total_cost:.4f}")
            print(f"   • Embedding model: {embedder.embedder.model}")
            print(f"   • Rate limit: 90 RPM")
            print()
            
            # Only ask for confirmation on fresh start
            response = input("💰 Proceed with embedding generation? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("❌ Embedding generation cancelled.")
                return

        # Generate embeddings (automatically resumes from checkpoint)
        print("\n🔄 Starting embedding generation...")
        qa_pairs_with_embeddings = await embedder.generate_embeddings_batch(qa_pairs)

        # Save final results
        print("\n💾 Saving final results...")
        embedder.save_embeddings(qa_pairs_with_embeddings, format="pickle")
        embedder.save_embeddings(qa_pairs_with_embeddings, format="numpy")
        
        # Generate final report
        report = embedder.save_progress_report()
        
        print(f"\n✅ Embedding generation completed!")
        print(f"   • Total pairs processed: {report['total_pairs']:,}")
        print(f"   • Success rate: {report['success_rate']:.1f}%")
        print(f"   • Total cost: ${report['total_cost_usd']:.4f}")
        print(f"   • Processing time: {report['processing_time_minutes']:.1f} minutes")
        print(f"   • Embeddings saved to: {embedder.embedding_output_dir}")
        
    except KeyboardInterrupt:
        print("\n⏸️  Process interrupted. Progress has been saved to checkpoint.")
        print("   Run the script again to resume from where you left off.")
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        print(f"\n❌ Error occurred: {e}")
        print("   Check logs for details. Progress has been saved to checkpoint.")


if __name__ == "__main__":
    asyncio.run(main()) 