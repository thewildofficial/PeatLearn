#!/usr/bin/env python3
"""
Mega Chunker for Ray Peat Corpus

Creates massive, context-preserving chunks optimized for million-token LLMs.
Focuses on preserving complete conversations, topics, and Ray Peat teachings.

Author: Aban Hasan
Date: 2025
"""

import re
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Chunking constants for million-token models
MAX_CHUNK_SIZE = 900000  # ~900K characters ≈ ~225K tokens (4:1 ratio)
MIN_CHUNK_SIZE = 100000  # Minimum meaningful chunk size
OVERLAP_SIZE = 10000     # Overlap between chunks to preserve context

@dataclass
class ChunkMetadata:
    """Metadata for each chunk."""
    chunk_id: int
    start_pos: int
    end_pos: int
    size_chars: int
    estimated_tokens: int
    break_type: str
    ray_peat_density: float
    key_topics: List[str]

class MegaChunker:
    """
    Advanced chunker for creating massive, coherent chunks from Ray Peat content.
    
    Optimized for:
    - Million-token context windows
    - Preserving complete conversations and topics
    - Maximizing Ray Peat signal density
    - Maintaining educational coherence
    """
    
    def __init__(self, max_chunk_size: int = MAX_CHUNK_SIZE):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = MIN_CHUNK_SIZE
        self.overlap_size = OVERLAP_SIZE
        
        # Ray Peat content patterns for intelligent chunking
        self.break_patterns = [
            # Highest priority - clear section breaks
            (r'\n\n=+ [^=]+ =+\n\n', 'section_header', 1.0),
            (r'\n\n#{1,3} [^\n]+\n\n', 'markdown_header', 1.0),
            
            # High priority - speaker changes in conversations
            (r'\n\n\*\*RAY PEAT:\*\*', 'ray_peat_speaker', 0.9),
            (r'\n\n\*\*HOST:\*\*', 'host_speaker', 0.8),
            (r'\n\n\*\*CALLER:\*\*', 'caller_speaker', 0.7),
            
            # Medium priority - topic boundaries
            (r'\n\n[A-Z][A-Z\s]{15,50}\n\n', 'topic_header', 0.6),
            (r'\n\n\d+\.\s+[A-Z][^\n]{20,}\n\n', 'numbered_section', 0.6),
            
            # Lower priority - natural breaks
            (r'\.\s*\n\n[A-Z][a-z]', 'paragraph_sentence', 0.4),
            (r'\n\n[A-Z][a-z]', 'paragraph_break', 0.3),
            (r'\n\n', 'simple_break', 0.1)
        ]
        
        # Ray Peat topic keywords for content analysis
        self.ray_peat_keywords = {
            'hormones': ['thyroid', 'progesterone', 'estrogen', 'cortisol', 'testosterone', 'insulin'],
            'metabolism': ['mitochondria', 'glucose', 'glycogen', 'metabolism', 'energy', 'atp'],
            'nutrition': ['pufa', 'saturated fat', 'fructose', 'sucrose', 'coconut oil', 'milk'],
            'supplements': ['aspirin', 'niacinamide', 'vitamin e', 'pregnenolone', 'cynomel'],
            'mechanisms': ['serotonin', 'histamine', 'nitric oxide', 'carbon dioxide', 'lactate'],
            'health_markers': ['temperature', 'pulse', 'blood sugar', 'cholesterol', 'inflammation']
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 characters per token for English)."""
        return len(text) // 4
    
    def analyze_ray_peat_density(self, text: str) -> Tuple[float, List[str]]:
        """
        Analyze the density of Ray Peat content in the text.
        
        Returns:
            Tuple of (density_ratio, found_topics)
        """
        if not text:
            return 0.0, []
        
        text_lower = text.lower()
        total_keywords = sum(len(keywords) for keywords in self.ray_peat_keywords.values())
        found_keywords = 0
        found_topics = []
        
        for category, keywords in self.ray_peat_keywords.items():
            category_found = False
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords += 1
                    if not category_found:
                        found_topics.append(category)
                        category_found = True
        
        # Bonus for Ray Peat attribution
        attribution_patterns = [
            r'\*\*ray peat:?\*\*',
            r'ray peat (says?|explains?|discusses?)',
            r'dr\.?\s+ray peat',
            r'according to (dr\.?\s+)?ray peat'
        ]
        
        attribution_bonus = 0
        for pattern in attribution_patterns:
            if re.search(pattern, text_lower):
                attribution_bonus += 0.1
        
        density = min((found_keywords / total_keywords) + attribution_bonus, 1.0)
        return density, found_topics
    
    def find_optimal_break_point(self, text: str, ideal_pos: int, search_radius: int = 50000) -> Tuple[int, str]:
        """
        Find the optimal break point near the ideal position.
        
        Args:
            text: Text to analyze
            ideal_pos: Ideal position for the break
            search_radius: How far to search from ideal position
            
        Returns:
            Tuple of (break_position, break_type)
        """
        start_search = max(0, ideal_pos - search_radius)
        end_search = min(len(text), ideal_pos + search_radius)
        search_text = text[start_search:end_search]
        
        best_break = ideal_pos
        best_priority = -1
        best_type = 'forced_break'
        
        # Search for break patterns in priority order
        for pattern, break_type, priority in self.break_patterns:
            matches = list(re.finditer(pattern, search_text))
            
            if matches and priority > best_priority:
                # Choose the match closest to ideal position
                ideal_relative = ideal_pos - start_search
                best_match = min(matches, key=lambda m: abs(m.start() - ideal_relative))
                
                candidate_pos = start_search + best_match.start()
                
                # Ensure we don't create too small chunks
                if candidate_pos > self.min_chunk_size and candidate_pos < len(text) - self.min_chunk_size:
                    best_break = candidate_pos
                    best_priority = priority
                    best_type = break_type
        
        logger.debug(f"Break point: pos {best_break}, type: {best_type}")
        return best_break, best_type
    
    def create_mega_chunks(self, text: str, preserve_overlap: bool = True) -> List[ChunkMetadata]:
        """
        Create massive chunks optimized for million-token processing.
        
        Args:
            text: Input text to chunk
            preserve_overlap: Whether to include overlap between chunks
            
        Returns:
            List of ChunkMetadata objects
        """
        if not text or len(text) < self.min_chunk_size:
            logger.warning("Text too short for meaningful chunking")
            return []
        
        chunks = []
        current_pos = 0
        chunk_id = 0
        
        logger.info(f"Creating mega-chunks from {len(text):,} characters")
        
        while current_pos < len(text):
            # Calculate ideal chunk end position
            ideal_end = current_pos + self.max_chunk_size
            
            # If this would be the last chunk, just take everything
            if ideal_end >= len(text) - self.min_chunk_size:
                chunk_end = len(text)
                break_type = 'end_of_text'
            else:
                # Find optimal break point
                chunk_end, break_type = self.find_optimal_break_point(text, ideal_end)
            
            # Extract chunk text
            chunk_text = text[current_pos:chunk_end]
            
            # Analyze chunk content
            ray_peat_density, topics = self.analyze_ray_peat_density(chunk_text)
            
            # Create chunk metadata
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                start_pos=current_pos,
                end_pos=chunk_end,
                size_chars=len(chunk_text),
                estimated_tokens=self.estimate_tokens(chunk_text),
                break_type=break_type,
                ray_peat_density=ray_peat_density,
                key_topics=topics
            )
            
            chunks.append(chunk_metadata)
            
            logger.info(f"Chunk {chunk_id}: {len(chunk_text):,} chars, "
                       f"~{chunk_metadata.estimated_tokens:,} tokens, "
                       f"density: {ray_peat_density:.3f}, "
                       f"topics: {len(topics)}, "
                       f"break: {break_type}")
            
            # Move to next chunk position
            if preserve_overlap and chunk_end < len(text):
                # Overlap with previous chunk to preserve context
                current_pos = max(chunk_end - self.overlap_size, current_pos + 1)
            else:
                current_pos = chunk_end
            
            chunk_id += 1
            
            # Safety check to prevent infinite loops
            if chunk_id > 1000:
                logger.error("Too many chunks created - stopping to prevent infinite loop")
                break
        
        # Add final chunk if needed
        if current_pos < len(text):
            final_text = text[current_pos:]
            if len(final_text) >= self.min_chunk_size:
                ray_peat_density, topics = self.analyze_ray_peat_density(final_text)
                
                final_chunk = ChunkMetadata(
                    chunk_id=chunk_id,
                    start_pos=current_pos,
                    end_pos=len(text),
                    size_chars=len(final_text),
                    estimated_tokens=self.estimate_tokens(final_text),
                    break_type='final_chunk',
                    ray_peat_density=ray_peat_density,
                    key_topics=topics
                )
                chunks.append(final_chunk)
        
        # Log summary
        total_chunks = len(chunks)
        avg_size = sum(c.size_chars for c in chunks) / max(total_chunks, 1)
        avg_tokens = sum(c.estimated_tokens for c in chunks) / max(total_chunks, 1)
        avg_density = sum(c.ray_peat_density for c in chunks) / max(total_chunks, 1)
        
        logger.info(f"Created {total_chunks} mega-chunks:")
        logger.info(f"  Average size: {avg_size:,.0f} chars (~{avg_tokens:,.0f} tokens)")
        logger.info(f"  Average Ray Peat density: {avg_density:.3f}")
        logger.info(f"  Total coverage: {sum(c.size_chars for c in chunks):,} chars")
        
        return chunks
    
    def extract_chunk_text(self, text: str, chunk_metadata: ChunkMetadata) -> str:
        """Extract the actual text for a chunk."""
        return text[chunk_metadata.start_pos:chunk_metadata.end_pos]
    
    def save_chunks(self, text: str, chunks: List[ChunkMetadata], output_dir: Path, base_filename: str):
        """
        Save chunks to individual files with metadata.
        
        Args:
            text: Original text
            chunks: List of chunk metadata
            output_dir: Directory to save chunks
            base_filename: Base name for chunk files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual chunks
        for chunk in chunks:
            chunk_text = self.extract_chunk_text(text, chunk)
            
            # Create chunk filename
            chunk_filename = f"{base_filename}_chunk_{chunk.chunk_id:03d}.txt"
            chunk_path = output_dir / chunk_filename
            
            # Save chunk text
            with open(chunk_path, 'w', encoding='utf-8') as f:
                # Add metadata header
                f.write(f"# Mega-Chunk {chunk.chunk_id}\n")
                f.write(f"# Size: {chunk.size_chars:,} chars (~{chunk.estimated_tokens:,} tokens)\n")
                f.write(f"# Ray Peat Density: {chunk.ray_peat_density:.3f}\n")
                f.write(f"# Topics: {', '.join(chunk.key_topics)}\n")
                f.write(f"# Break Type: {chunk.break_type}\n")
                f.write(f"# Position: {chunk.start_pos:,}-{chunk.end_pos:,}\n")
                f.write("\n" + "="*80 + "\n\n")
                f.write(chunk_text)
        
        # Save chunk metadata
        metadata_file = output_dir / f"{base_filename}_chunks_metadata.json"
        import json
        
        metadata = {
            'base_filename': base_filename,
            'total_chunks': len(chunks),
            'chunking_config': {
                'max_chunk_size': self.max_chunk_size,
                'min_chunk_size': self.min_chunk_size,
                'overlap_size': self.overlap_size
            },
            'chunks': [
                {
                    'chunk_id': c.chunk_id,
                    'start_pos': c.start_pos,
                    'end_pos': c.end_pos,
                    'size_chars': c.size_chars,
                    'estimated_tokens': c.estimated_tokens,
                    'break_type': c.break_type,
                    'ray_peat_density': c.ray_peat_density,
                    'key_topics': c.key_topics
                }
                for c in chunks
            ]
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {len(chunks)} chunks to {output_dir}")

def chunk_file(input_file: Path, output_dir: Path, max_chunk_size: int = MAX_CHUNK_SIZE) -> List[ChunkMetadata]:
    """
    Chunk a single file into mega-chunks.
    
    Args:
        input_file: Path to input file
        output_dir: Directory for output chunks
        max_chunk_size: Maximum size per chunk
        
    Returns:
        List of chunk metadata
    """
    logger.info(f"Chunking file: {input_file}")
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    # Create chunker
    chunker = MegaChunker(max_chunk_size=max_chunk_size)
    
    # Create chunks
    chunks = chunker.create_mega_chunks(text)
    
    # Save chunks
    base_filename = input_file.stem
    chunker.save_chunks(text, chunks, output_dir, base_filename)
    
    return chunks

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create mega-chunks from Ray Peat content')
    parser.add_argument('--input-file', required=True, help='Input file to chunk')
    parser.add_argument('--output-dir', required=True, help='Output directory for chunks')
    parser.add_argument('--max-chunk-size', type=int, default=MAX_CHUNK_SIZE, 
                       help='Maximum chunk size in characters')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Chunk the file
    chunks = chunk_file(
        input_file=Path(args.input_file),
        output_dir=Path(args.output_dir),
        max_chunk_size=args.max_chunk_size
    )
    
    print(f"Created {len(chunks)} mega-chunks") 