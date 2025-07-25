#!/usr/bin/env python3
"""
Unified Ray Peat Signal Processor

A streamlined pipeline that maximizes signal extraction from the Ray Peat corpus
by leveraging million-token context windows for massive chunk processing.

Strategy:
1. Rules-based preprocessing for basic cleaning
2. If insufficient signal detected, escalate to AI enhancement
3. Create massive chunks (up to 1M tokens) for maximum context preservation
4. Focus purely on Ray Peat signal extraction

Author: Aban Hasan
Date: 2025
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import google.generativeai as genai
from typing import Dict, List, Tuple, Optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, that's okay
import re
from dataclasses import dataclass

# Import existing cleaners
import rules_based_cleaners as rbc
import ai_powered_cleaners as aic

# Environment variables loaded above

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for million-token processing
MAX_CHUNK_SIZE = 800000  # ~800K characters ≈ ~200K tokens (conservative estimate)
MIN_SIGNAL_THRESHOLD = 0.3  # Minimum Ray Peat signal ratio to proceed
TIER1_THRESHOLDS = {
    'textual_fidelity_score': 4,
    'document_atomicity_score': 5,
    'semantic_noise_score': 5
}

@dataclass
class ProcessingResult:
    """Result of processing a single file."""
    success: bool
    original_size: int
    processed_size: int
    signal_ratio: float
    processing_method: str  # 'rules_only', 'ai_enhanced'
    key_topics: List[str]
    error_message: Optional[str] = None

class UnifiedSignalProcessor:
    """Unified processor that maximizes Ray Peat signal extraction."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the unified processor."""
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.ai_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("AI enhancement available")
        else:
            self.ai_model = None
            logger.warning("No API key provided - AI enhancement disabled")
        
        self.stats = {
            'total_files': 0,
            'rules_only': 0,
            'ai_enhanced': 0,
            'failed': 0,
            'total_input_size': 0,
            'total_output_size': 0
        }
    
    def detect_ray_peat_signal(self, content: str) -> Tuple[float, List[str]]:
        """
        Detect Ray Peat signal strength and key topics using pattern matching.
        
        Returns:
            Tuple of (signal_ratio, key_topics)
        """
        if not content:
            return 0.0, []
        
        # Ray Peat signature terms and concepts
        ray_peat_indicators = {
            'core_concepts': [
                'thyroid', 'progesterone', 'estrogen', 'cortisol', 'pufa',
                'metabolism', 'bioenergetic', 'mitochondria', 'glycogen',
                'glucose', 'fructose', 'saturated fat', 'coconut oil'
            ],
            'mechanisms': [
                'oxidative metabolism', 'respiratory quotient', 'metabolic rate',
                'hormone regulation', 'stress response', 'inflammation',
                'serotonin', 'histamine', 'nitric oxide', 'carbon dioxide'
            ],
            'practical': [
                'temperature', 'pulse rate', 'milk', 'cheese', 'orange juice',
                'aspirin', 'niacinamide', 'vitamin e', 'pregnenolone'
            ]
        }
        
        content_lower = content.lower()
        total_indicators = sum(len(category) for category in ray_peat_indicators.values())
        found_indicators = 0
        found_topics = []
        
        for category, terms in ray_peat_indicators.items():
            for term in terms:
                if term in content_lower:
                    found_indicators += 1
                    found_topics.append(term)
        
        # Check for Ray Peat attribution
        attribution_patterns = [
            r'\*\*ray peat:?\*\*',
            r'ray peat says?',
            r'according to ray peat',
            r'dr\.? ray peat',
            r'peat explains?'
        ]
        
        attribution_score = 0
        for pattern in attribution_patterns:
            if re.search(pattern, content_lower):
                attribution_score += 0.2
        
        # Calculate signal ratio
        base_ratio = found_indicators / total_indicators
        signal_ratio = min(base_ratio + attribution_score, 1.0)
        
        return signal_ratio, list(set(found_topics))
    
    def create_mega_chunks(self, content: str) -> List[Tuple[str, int, int]]:
        """
        Create massive chunks optimized for million-token context windows.
        
        Prioritizes:
        1. Preserving complete conversations/sections
        2. Maintaining context boundaries
        3. Maximizing chunk size for better AI understanding
        """
        if len(content) <= MAX_CHUNK_SIZE:
            return [(content, 0, len(content))]
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(content):
            # Calculate potential chunk end
            chunk_end = min(current_pos + MAX_CHUNK_SIZE, len(content))
            
            # If not at the end, find optimal break point
            if chunk_end < len(content):
                # Look for natural breaks in the last 20% of chunk
                search_start = max(current_pos, chunk_end - MAX_CHUNK_SIZE // 5)
                
                # Priority order for break points
                break_patterns = [
                    (r'\n\n=+ .+ =+\n\n', 'section_header'),  # Section headers
                    (r'\n\n\*\*RAY PEAT:\*\*', 'ray_peat_speaker'),  # Ray Peat speaking
                    (r'\n\n\*\*HOST:\*\*', 'host_speaker'),   # Host speaking
                    (r'\n\n[A-Z][A-Z ]{20,}\n\n', 'topic_header'),  # Topic headers
                    (r'\.\s*\n\n[A-Z]', 'paragraph_end'),    # End of paragraph + new sentence
                    (r'\n\n', 'paragraph_break')             # Any paragraph break
                ]
                
                best_break = chunk_end
                best_priority = len(break_patterns)
                
                for priority, (pattern, break_type) in enumerate(break_patterns):
                    matches = list(re.finditer(pattern, content[search_start:chunk_end]))
                    if matches and priority < best_priority:
                        # Use the last match of highest priority type
                        best_break = search_start + matches[-1].start()
                        best_priority = priority
                        logger.debug(f"Found {break_type} break at pos {best_break}")
                        break
                
                chunk_end = best_break
            
            chunk_text = content[current_pos:chunk_end].strip()
            if chunk_text:  # Only add non-empty chunks
                chunks.append((chunk_text, current_pos, chunk_end))
                logger.info(f"Created mega-chunk: {len(chunk_text):,} chars (pos {current_pos:,}-{chunk_end:,})")
            
            current_pos = chunk_end
        
        return chunks
    
    def ai_enhance_signal(self, content: str) -> Dict:
        """
        Use AI to enhance Ray Peat signal extraction with massive context windows.
        """
        if not self.ai_model:
            raise ValueError("AI model not available")
        
        # Create mega chunks for processing
        chunks = self.create_mega_chunks(content)
        enhanced_parts = []
        
        # Enhanced prompt for mega-chunk processing
        MEGA_CHUNK_PROMPT = """You are extracting Ray Peat's bioenergetic wisdom from a large document section.

Your mission: Extract MAXIMUM SIGNAL from Ray Peat's teachings while removing all noise.

REMOVE COMPLETELY:
- All advertisements, promotions, and sponsorship content
- Host/interviewer introductions and show logistics
- Social talk, pleasantries, and off-topic conversations  
- Website mentions, contact info, and commercial content
- Technical difficulties and caller management
- Repetitive filler and meandering discussions

PRESERVE AND ENHANCE:
- All Ray Peat explanations, insights, and scientific discussions
- Core bioenergetic principles and mechanisms
- Practical health recommendations and protocols
- Research citations and scientific evidence
- Relevant questions that provide context for Ray Peat's answers

FORMAT OUTPUT AS:
**RAY PEAT:** [His exact teachings and explanations - preserve fully]
**CONTEXT:** [Only essential questions/setup that frame Ray Peat's responses]

MAINTAIN:
- Complete scientific explanations (don't truncate)
- Technical terminology and precise language
- Logical flow of concepts
- All practical recommendations

Your goal is to create a compendious, pure-signal educational resource focused entirely on Ray Peat's bioenergetic teachings.

Document section to process:
{content}
"""
        
        for i, (chunk, start_pos, end_pos) in enumerate(chunks):
            logger.info(f"AI processing mega-chunk {i+1}/{len(chunks)} ({len(chunk):,} chars)")
            
            try:
                prompt = MEGA_CHUNK_PROMPT.format(content=chunk)
                
                response = self.ai_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=8192,
                        temperature=0.1,
                        candidate_count=1
                    )
                )
                
                enhanced_content = response.text.strip()
                
                # Quality validation
                if len(enhanced_content) < 200:
                    logger.warning(f"Chunk {i+1} produced very short output")
                    continue
                
                # Check for Ray Peat content preservation
                signal_ratio, topics = self.detect_ray_peat_signal(enhanced_content)
                if signal_ratio < 0.1:
                    logger.warning(f"Chunk {i+1} may have lost Ray Peat signal")
                    continue
                
                enhanced_parts.append(enhanced_content)
                logger.info(f"Chunk {i+1} processed: {signal_ratio:.2f} signal ratio, {len(topics)} topics")
                
            except Exception as e:
                logger.error(f"AI processing failed for chunk {i+1}: {e}")
                continue
        
        if not enhanced_parts:
            raise ValueError("AI enhancement failed - no valid output generated")
        
        # Combine enhanced parts
        combined_content = "\n\n" + "="*50 + "\n\n".join(enhanced_parts)
        final_signal_ratio, final_topics = self.detect_ray_peat_signal(combined_content)
        
        return {
            'enhanced_content': combined_content,
            'signal_ratio': final_signal_ratio,
            'key_topics': final_topics,
            'chunks_processed': len(chunks),
            'chunks_successful': len(enhanced_parts)
        }
    
    def process_file(self, file_path: Path, tier_info: Optional[Dict] = None) -> ProcessingResult:
        """
        Process a single file with unified rules + AI approach.
        
        Strategy:
        1. Apply rules-based cleaning first
        2. Assess signal quality
        3. If signal too low, enhance with AI
        4. Return best result
        """
        try:
            logger.info(f"Processing: {file_path.name}")
            
            # Read original content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_content = f.read()
            
            original_size = len(original_content)
            self.stats['total_input_size'] += original_size
            
            # Step 1: Rules-based cleaning
            if file_path.suffix.lower() == '.html':
                rules_cleaned = rbc.clean_html(str(file_path))
            else:
                rules_cleaned = original_content
            
            # Apply standard rules cleaning
            rules_cleaned = rbc.normalize_whitespace(rules_cleaned)
            rules_cleaned = rbc.remove_known_artifacts(rules_cleaned, str(file_path))
            
            # Step 2: Assess signal quality
            signal_ratio, topics = self.detect_ray_peat_signal(rules_cleaned)
            logger.info(f"Rules-only signal ratio: {signal_ratio:.3f}, Topics: {len(topics)}")
            
            # Step 3: Decide on processing strategy
            processing_method = 'rules_only'
            final_content = rules_cleaned
            
            # Use tier info if available to make smarter decisions
            needs_ai_enhancement = False
            if tier_info:
                # Low quality files automatically get AI enhancement
                if (tier_info.get('textual_fidelity_score', 10) < TIER1_THRESHOLDS['textual_fidelity_score'] or
                    tier_info.get('semantic_noise_score', 10) < TIER1_THRESHOLDS['semantic_noise_score']):
                    needs_ai_enhancement = True
                    logger.info("Tier analysis indicates AI enhancement needed")
            
            # Also enhance if signal ratio is too low
            if signal_ratio < MIN_SIGNAL_THRESHOLD:
                needs_ai_enhancement = True
                logger.info(f"Signal ratio {signal_ratio:.3f} below threshold {MIN_SIGNAL_THRESHOLD}")
            
            # Step 4: AI enhancement if needed and available
            if needs_ai_enhancement and self.ai_model and len(rules_cleaned) > 500:
                try:
                    logger.info("Applying AI enhancement...")
                    ai_result = self.ai_enhance_signal(rules_cleaned)
                    
                    # Use AI result if it's significantly better
                    if ai_result['signal_ratio'] > signal_ratio * 1.2:  # 20% improvement threshold
                        final_content = ai_result['enhanced_content']
                        signal_ratio = ai_result['signal_ratio']
                        topics = ai_result['key_topics']
                        processing_method = 'ai_enhanced'
                        logger.info(f"AI enhancement successful: {signal_ratio:.3f} signal ratio")
                    else:
                        logger.info("AI enhancement didn't significantly improve signal")
                        
                except Exception as e:
                    logger.error(f"AI enhancement failed: {e}")
                    # Fall back to rules-only result
            
            # Update stats
            self.stats[processing_method] += 1
            processed_size = len(final_content)
            self.stats['total_output_size'] += processed_size
            
            return ProcessingResult(
                success=True,
                original_size=original_size,
                processed_size=processed_size,
                signal_ratio=signal_ratio,
                processing_method=processing_method,
                key_topics=topics
            )
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            self.stats['failed'] += 1
            return ProcessingResult(
                success=False,
                original_size=0,
                processed_size=0,
                signal_ratio=0.0,
                processing_method='failed',
                key_topics=[],
                error_message=str(e)
            )
    
    def process_corpus(self, 
                      input_dir: str,
                      output_dir: str,
                      analysis_file: Optional[str] = None,
                      limit: Optional[int] = None) -> Dict:
        """
        Process entire corpus with unified approach.
        
        Args:
            input_dir: Directory containing raw files
            output_dir: Directory for processed output
            analysis_file: Optional CSV with tier analysis
            limit: Optional limit on number of files to process
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load tier analysis if available
        tier_data = {}
        if analysis_file and Path(analysis_file).exists():
            try:
                df = pd.read_csv(analysis_file)
                for _, row in df.iterrows():
                    file_path = row['file_path']
                    tier_data[file_path] = {
                        'textual_fidelity_score': row.get('textual_fidelity_score', 10),
                        'semantic_noise_score': row.get('semantic_noise_score', 10),
                        'document_atomicity_score': row.get('document_atomicity_score', 10)
                    }
                logger.info(f"Loaded tier analysis for {len(tier_data)} files")
            except Exception as e:
                logger.warning(f"Could not load tier analysis: {e}")
        
        # Find all files to process
        file_patterns = ['*.txt', '*.html', '*.md', '*.pdf']
        all_files = []
        for pattern in file_patterns:
            all_files.extend(input_path.rglob(pattern))
        
        if limit:
            all_files = all_files[:limit]
        
        logger.info(f"Found {len(all_files)} files to process")
        self.stats['total_files'] = len(all_files)
        
        # Process files
        results = []
        metadata = {'processing_timestamp': datetime.now().isoformat(), 'files': []}
        
        for i, file_path in enumerate(all_files, 1):
            logger.info(f"Processing file {i}/{len(all_files)}: {file_path.name}")
            
            # Get tier info for this file
            rel_path = str(file_path.relative_to(input_path.parent))
            tier_info = tier_data.get(rel_path)
            
            # Process the file
            result = self.process_file(file_path, tier_info)
            results.append(result)
            
            if result.success:
                # Save processed content
                output_file = output_path / f"{file_path.stem}_processed.txt"
                
                # Get the actual processed content from the processing result
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as orig_f:
                    original = orig_f.read()
                
                if result.processing_method == 'rules_only':
                    # Apply rules cleaning
                    if file_path.suffix.lower() == '.html':
                        processed = rbc.clean_html(str(file_path))
                    else:
                        processed = original
                    processed = rbc.normalize_whitespace(processed)
                    processed = rbc.remove_known_artifacts(processed, str(file_path))
                else:
                    # For AI-enhanced, we need to store the content in the result object
                    # For now, re-process to get the content (this is inefficient but works)
                    if file_path.suffix.lower() == '.html':
                        rules_cleaned = rbc.clean_html(str(file_path))
                    else:
                        rules_cleaned = original
                    rules_cleaned = rbc.normalize_whitespace(rules_cleaned)
                    rules_cleaned = rbc.remove_known_artifacts(rules_cleaned, str(file_path))
                    
                    # Re-run AI enhancement to get the content
                    try:
                        ai_result = self.ai_enhance_signal(rules_cleaned)
                        processed = ai_result['enhanced_content']
                    except Exception as e:
                        logger.error(f"Failed to get AI-enhanced content for saving: {e}")
                        processed = rules_cleaned  # Fall back to rules-only
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(processed)
                
                # Add to metadata
                metadata['files'].append({
                    'original_file': str(file_path),
                    'output_file': str(output_file),
                    'processing_method': result.processing_method,
                    'signal_ratio': result.signal_ratio,
                    'key_topics': result.key_topics,
                    'original_size': result.original_size,
                    'processed_size': result.processed_size,
                    'compression_ratio': result.processed_size / max(result.original_size, 1)
                })
        
        # Save metadata
        metadata_file = output_path / 'processing_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate summary
        successful_results = [r for r in results if r.success]
        avg_signal_ratio = sum(r.signal_ratio for r in successful_results) / max(len(successful_results), 1)
        
        summary = {
            'total_files': len(all_files),
            'successful': len(successful_results),
            'failed': len([r for r in results if not r.success]),
            'rules_only': self.stats['rules_only'],
            'ai_enhanced': self.stats['ai_enhanced'],
            'average_signal_ratio': avg_signal_ratio,
            'total_input_size_mb': self.stats['total_input_size'] / (1024*1024),
            'total_output_size_mb': self.stats['total_output_size'] / (1024*1024),
            'compression_ratio': self.stats['total_output_size'] / max(self.stats['total_input_size'], 1)
        }
        
        logger.info("Processing Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        return summary

def main():
    """Main entry point for unified signal processing."""
    parser = argparse.ArgumentParser(description='Unified Ray Peat Signal Processor')
    parser.add_argument('--input-dir', required=True, help='Input directory with raw files')
    parser.add_argument('--output-dir', required=True, help='Output directory for processed files')
    parser.add_argument('--analysis-file', help='Optional CSV file with tier analysis')
    parser.add_argument('--limit', type=int, help='Limit number of files to process')
    parser.add_argument('--api-key', help='Google API key for AI enhancement')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = UnifiedSignalProcessor(api_key=args.api_key)
    
    # Process corpus
    summary = processor.process_corpus(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        analysis_file=args.analysis_file,
        limit=args.limit
    )
    
    print("\nProcessing completed successfully!")
    print(f"Processed {summary['successful']}/{summary['total_files']} files")
    print(f"Average signal ratio: {summary['average_signal_ratio']:.3f}")
    print(f"Files using AI enhancement: {summary['ai_enhanced']}")

if __name__ == "__main__":
    main() 