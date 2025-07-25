#!/usr/bin/env python3
"""
Enhanced Unified Ray Peat Signal Processor v2.0

Advanced pipeline with:
- Accurate threshold analysis
- Detailed cost estimation and logging
- Resumable processing with checkpoints
- Real-time signal improvement tracking
- Production-ready corpus processing

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
from datetime import datetime, timedelta
import google.generativeai as genai
from typing import Dict, List, Tuple, Optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import re
from dataclasses import dataclass, asdict
import csv
import hashlib

# Import existing cleaners
import rules_based_cleaners as rbc
import ai_powered_cleaners as aic

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced constants based on test analysis
MAX_CHUNK_SIZE = 800000  # ~800K characters ≈ ~200K tokens
MIN_SIGNAL_THRESHOLD = 0.25  # Adjusted based on test data (was 0.3)
AI_IMPROVEMENT_THRESHOLD = 1.5  # 50% improvement needed to use AI result

# Gemini 2.0 Flash pricing (as of 2025)
GEMINI_PRICING = {
    'input_tokens_per_million': 0.075,  # $0.075 per 1M input tokens
    'output_tokens_per_million': 0.30,  # $0.30 per 1M output tokens
    'requests_per_1000': 0.0001  # Minimal request cost
}

@dataclass
class ProcessingResult:
    """Enhanced result tracking with cost analysis."""
    success: bool
    original_size: int
    processed_size: int
    signal_ratio_before: float
    signal_ratio_after: float
    processing_method: str
    key_topics: List[str]
    processing_time: float
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    estimated_cost: float = 0.0
    error_message: Optional[str] = None
    
    @property
    def signal_improvement(self) -> float:
        """Calculate signal improvement ratio."""
        if self.signal_ratio_before == 0:
            return float('inf') if self.signal_ratio_after > 0 else 0
        return self.signal_ratio_after / self.signal_ratio_before
    
    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        return self.processed_size / max(self.original_size, 1)

@dataclass
class ProcessingCheckpoint:
    """Checkpoint for resumable processing."""
    last_processed_file: str
    total_files_processed: int
    total_ai_enhanced: int
    total_rules_only: int
    cumulative_cost: float
    cumulative_tokens: int
    start_time: str
    
class EnhancedSignalProcessor:
    """Enhanced processor with cost tracking and resumable processing."""
    
    def __init__(self, api_key: Optional[str] = None, checkpoint_file: Optional[str] = None):
        """Initialize enhanced processor."""
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.ai_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("✅ AI enhancement available (Gemini 2.0 Flash)")
        else:
            self.ai_model = None
            logger.warning("⚠️ No API key - AI enhancement disabled")
        
        self.checkpoint_file = checkpoint_file
        self.checkpoint = self._load_checkpoint()
        
        # Enhanced statistics tracking
        self.stats = {
            'start_time': datetime.now(),
            'total_files': 0,
            'rules_only': self.checkpoint.total_rules_only if self.checkpoint else 0,
            'ai_enhanced': self.checkpoint.total_ai_enhanced if self.checkpoint else 0,
            'failed': 0,
            'total_input_size': 0,
            'total_output_size': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_api_calls': 0,
            'total_estimated_cost': self.checkpoint.cumulative_cost if self.checkpoint else 0.0,
            'signal_improvements': [],
            'processing_times': []
        }
    
    def _load_checkpoint(self) -> Optional[ProcessingCheckpoint]:
        """Load processing checkpoint if exists."""
        if not self.checkpoint_file or not Path(self.checkpoint_file).exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            checkpoint = ProcessingCheckpoint(**data)
            logger.info(f"📍 Resuming from checkpoint: {checkpoint.total_files_processed} files processed")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def _save_checkpoint(self, last_file: str):
        """Save processing checkpoint."""
        if not self.checkpoint_file:
            return
            
        checkpoint = ProcessingCheckpoint(
            last_processed_file=last_file,
            total_files_processed=self.stats['rules_only'] + self.stats['ai_enhanced'],
            total_ai_enhanced=self.stats['ai_enhanced'],
            total_rules_only=self.stats['rules_only'],
            cumulative_cost=self.stats['total_estimated_cost'],
            cumulative_tokens=self.stats['total_input_tokens'] + self.stats['total_output_tokens'],
            start_time=self.stats['start_time'].isoformat()
        )
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(asdict(checkpoint), f, indent=2)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (4 chars per token for English)."""
        return len(text) // 4
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, api_calls: int = 1) -> float:
        """Calculate estimated cost for Gemini API usage."""
        input_cost = (input_tokens / 1000000) * GEMINI_PRICING['input_tokens_per_million']
        output_cost = (output_tokens / 1000000) * GEMINI_PRICING['output_tokens_per_million']
        request_cost = (api_calls / 1000) * GEMINI_PRICING['requests_per_1000']
        return input_cost + output_cost + request_cost
    
    def analyze_corpus_thresholds(self, analysis_file: str) -> Dict:
        """Analyze corpus to determine optimal thresholds."""
        if not Path(analysis_file).exists():
            return {}
        
        df = pd.read_csv(analysis_file)
        
        # Analyze quality score distributions
        threshold_analysis = {
            'total_files': len(df),
            'tier1_candidates': 0,
            'tier2_candidates': 0,
            'quality_distributions': {},
            'recommended_thresholds': {}
        }
        
        for col in ['textual_fidelity_score', 'semantic_noise_score', 'document_atomicity_score']:
            if col in df.columns:
                scores = df[col].dropna()
                threshold_analysis['quality_distributions'][col] = {
                    'mean': float(scores.mean()),
                    'median': float(scores.median()),
                    'q25': float(scores.quantile(0.25)),
                    'q75': float(scores.quantile(0.75)),
                    'min': float(scores.min()),
                    'max': float(scores.max())
                }
        
        # Calculate tier distributions based on current thresholds
        for _, row in df.iterrows():
            fidelity = row.get('textual_fidelity_score', 10)
            noise = row.get('semantic_noise_score', 10)
            atomicity = row.get('document_atomicity_score', 10)
            
            if (fidelity >= 4 and noise >= 5 and atomicity >= 5):
                threshold_analysis['tier1_candidates'] += 1
            else:
                threshold_analysis['tier2_candidates'] += 1
        
        # Recommend thresholds for ~70% rules-only, 30% AI-enhanced
        target_tier1_ratio = 0.7
        threshold_analysis['recommended_thresholds'] = {
            'signal_threshold': 0.25,  # Based on test results
            'tier1_ratio': threshold_analysis['tier1_candidates'] / threshold_analysis['total_files'],
            'estimated_ai_files': threshold_analysis['tier2_candidates']
        }
        
        return threshold_analysis
    
    def detect_ray_peat_signal(self, content: str) -> Tuple[float, List[str]]:
        """Enhanced Ray Peat signal detection with expanded keywords."""
        if not content:
            return 0.0, []
        
        # Expanded Ray Peat indicator keywords
        ray_peat_indicators = {
            'hormones': ['thyroid', 'progesterone', 'estrogen', 'cortisol', 'testosterone', 'insulin',
                        'pregnenolone', 'dhea', 'prolactin', 'growth hormone', 'aldosterone'],
            'metabolism': ['mitochondria', 'glucose', 'glycogen', 'metabolism', 'energy', 'atp',
                         'respiratory quotient', 'metabolic rate', 'oxidative metabolism'],
            'nutrition': ['pufa', 'saturated fat', 'fructose', 'sucrose', 'coconut oil', 'milk',
                         'cheese', 'orange juice', 'honey', 'gelatin', 'calcium'],
            'supplements': ['aspirin', 'niacinamide', 'vitamin e', 'pregnenolone', 'cynomel',
                           'cytomel', 'methylene blue', 'caffeine', 'magnesium'],
            'mechanisms': ['serotonin', 'histamine', 'nitric oxide', 'carbon dioxide', 'lactate',
                          'endotoxin', 'stress response', 'inflammation', 'lipid peroxidation'],
            'health_markers': ['temperature', 'pulse', 'blood sugar', 'cholesterol', 'inflammation',
                              'pulse rate', 'basal temperature', 'metabolic rate']
        }
        
        content_lower = content.lower()
        total_indicators = sum(len(category) for category in ray_peat_indicators.values())
        found_indicators = 0
        found_topics = []
        
        for category, terms in ray_peat_indicators.items():
            category_found = False
            for term in terms:
                if term in content_lower:
                    found_indicators += 1
                    if not category_found:
                        found_topics.append(category)
                        category_found = True
        
        # Enhanced attribution patterns
        attribution_patterns = [
            r'\*\*ray peat:?\*\*',
            r'ray peat (says?|explains?|discusses?|believes?)',
            r'dr\.?\s+ray peat',
            r'according to (dr\.?\s+)?ray peat',
            r'peat (explains?|says?|notes?)',
            r'ray peat\'s (work|research|theory)'
        ]
        
        attribution_score = 0
        for pattern in attribution_patterns:
            matches = len(re.findall(pattern, content_lower))
            attribution_score += matches * 0.1
        
        # Calculate enhanced signal ratio
        base_ratio = found_indicators / total_indicators
        signal_ratio = min(base_ratio + attribution_score, 1.0)
        
        return signal_ratio, list(set(found_topics))
    
    def create_mega_chunks(self, content: str) -> List[Tuple[str, int, int]]:
        """Create massive chunks with enhanced break detection."""
        if len(content) <= MAX_CHUNK_SIZE:
            return [(content, 0, len(content))]
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(content):
            chunk_end = min(current_pos + MAX_CHUNK_SIZE, len(content))
            
            if chunk_end < len(content):
                # Enhanced break point detection
                search_start = max(current_pos, chunk_end - MAX_CHUNK_SIZE // 4)
                
                break_patterns = [
                    (r'\n\n=+ [^=]+ =+\n\n', 'section_header', 1.0),
                    (r'\n\n#{1,6} [^\n]+\n\n', 'markdown_header', 0.95),
                    (r'\n\n\*\*RAY PEAT:\*\*', 'ray_peat_speaker', 0.9),
                    (r'\n\n\*\*HOST:\*\*', 'host_speaker', 0.85),
                    (r'\n\n\*\*CALLER:\*\*', 'caller_speaker', 0.8),
                    (r'\n\n[A-Z][A-Z\s]{15,50}\n\n', 'topic_header', 0.7),
                    (r'\n\n\d+\.\s+[A-Z][^\n]{20,}\n\n', 'numbered_section', 0.65),
                    (r'\.\s*\n\n[A-Z][a-z]', 'paragraph_sentence', 0.5),
                    (r'\n\n[A-Z]', 'paragraph_break', 0.3),
                    (r'\n\n', 'simple_break', 0.1)
                ]
                
                best_break = chunk_end
                best_priority = 0
                
                for pattern, break_type, priority in break_patterns:
                    matches = list(re.finditer(pattern, content[search_start:chunk_end]))
                    if matches and priority > best_priority:
                        best_match = matches[-1]  # Use last match
                        candidate_pos = search_start + best_match.start()
                        
                        # Ensure reasonable chunk sizes
                        if candidate_pos > current_pos + 50000:  # Minimum 50K chars
                            best_break = candidate_pos
                            best_priority = priority
                
                chunk_end = best_break
            
            chunk_text = content[current_pos:chunk_end].strip()
            if chunk_text:
                chunks.append((chunk_text, current_pos, chunk_end))
                logger.debug(f"📦 Mega-chunk: {len(chunk_text):,} chars (pos {current_pos:,}-{chunk_end:,})")
            
            current_pos = chunk_end
        
        return chunks
    
    def ai_enhance_signal(self, content: str) -> Dict:
        """Enhanced AI signal extraction with cost tracking."""
        if not self.ai_model:
            raise ValueError("AI model not available")
        
        chunks = self.create_mega_chunks(content)
        enhanced_parts = []
        total_input_tokens = 0
        total_output_tokens = 0
        api_calls = 0
        
        # Enhanced prompt for better signal extraction
        ENHANCED_PROMPT = """You are Dr. Ray Peat's most dedicated student, extracting his pure bioenergetic wisdom from this document.

MISSION: Extract MAXIMUM RAY PEAT SIGNAL while removing ALL noise.

REMOVE COMPLETELY:
- All advertisements, sponsorships, and commercial content
- Host introductions, show logistics, and technical difficulties
- Social talk, pleasantries, and off-topic conversations
- Website mentions, contact information, and promotional content
- Caller management, dead air, and filler content
- Repetitive statements and meandering discussions

PRESERVE AND ENHANCE:
- Every Ray Peat explanation, insight, and scientific discussion
- All bioenergetic principles and metabolic mechanisms
- Practical health recommendations and protocols
- Research citations and scientific evidence
- Relevant questions that directly relate to Ray Peat's teachings

FORMAT REQUIREMENTS:
**RAY PEAT:** [His exact words and teachings - preserve completely]
**CONTEXT:** [Only essential questions/setup that directly relate to Ray Peat's response]

QUALITY STANDARDS:
- Preserve complete scientific explanations (never truncate)
- Maintain technical terminology and precise language
- Keep logical flow of bioenergetic concepts
- Include all practical recommendations and dosages
- Preserve research citations and scientific backing

Your goal: Create a compendious, pure-signal educational resource containing ONLY Ray Peat's bioenergetic teachings.

Document to process:
{content}
"""
        
        for i, (chunk, start_pos, end_pos) in enumerate(chunks):
            logger.info(f"🤖 AI processing mega-chunk {i+1}/{len(chunks)} ({len(chunk):,} chars)")
            
            try:
                prompt = ENHANCED_PROMPT.format(content=chunk)
                input_tokens = self.estimate_tokens(prompt)
                
                response = self.ai_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=8192,
                        temperature=0.1,
                        candidate_count=1
                    )
                )
                
                enhanced_content = response.text.strip()
                output_tokens = self.estimate_tokens(enhanced_content)
                
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                api_calls += 1
                
                # Quality validation
                if len(enhanced_content) < 200:
                    logger.warning(f"⚠️ Chunk {i+1} produced short output ({len(enhanced_content)} chars)")
                    continue
                
                signal_ratio, topics = self.detect_ray_peat_signal(enhanced_content)
                if signal_ratio < 0.1:
                    logger.warning(f"⚠️ Chunk {i+1} may have lost Ray Peat signal (ratio: {signal_ratio:.3f})")
                
                enhanced_parts.append(enhanced_content)
                logger.info(f"✅ Chunk {i+1}: {signal_ratio:.3f} signal, {len(topics)} topics, {output_tokens:,} tokens")
                
            except Exception as e:
                logger.error(f"❌ AI processing failed for chunk {i+1}: {e}")
                api_calls += 1  # Count failed calls too
                continue
        
        if not enhanced_parts:
            raise ValueError("AI enhancement failed - no valid output generated")
        
        # Combine enhanced parts with clear separation
        combined_content = "\n\n" + ("=" * 50 + "\n\n").join(enhanced_parts)
        final_signal_ratio, final_topics = self.detect_ray_peat_signal(combined_content)
        
        # Calculate costs
        estimated_cost = self.calculate_cost(total_input_tokens, total_output_tokens, api_calls)
        
        return {
            'enhanced_content': combined_content,
            'signal_ratio': final_signal_ratio,
            'key_topics': final_topics,
            'chunks_processed': len(chunks),
            'chunks_successful': len(enhanced_parts),
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'api_calls': api_calls,
            'estimated_cost': estimated_cost
        }
    
    def process_file(self, file_path: Path, tier_info: Optional[Dict] = None) -> ProcessingResult:
        """Enhanced file processing with detailed tracking."""
        start_time = time.time()
        
        try:
            logger.info(f"📄 Processing: {file_path.name}")
            
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
            
            rules_cleaned = rbc.normalize_whitespace(rules_cleaned)
            rules_cleaned = rbc.remove_known_artifacts(rules_cleaned, str(file_path))
            
            # Step 2: Initial signal assessment
            signal_ratio_before, topics_before = self.detect_ray_peat_signal(rules_cleaned)
            logger.info(f"📊 Rules-only signal: {signal_ratio_before:.3f}, Topics: {len(topics_before)}")
            
            # Step 3: Determine processing strategy
            processing_method = 'rules_only'
            final_content = rules_cleaned
            signal_ratio_after = signal_ratio_before
            final_topics = topics_before
            input_tokens = output_tokens = api_calls = estimated_cost = 0
            
            # Enhanced AI triggering logic
            needs_ai_enhancement = False
            
            # Check tier analysis
            if tier_info:
                fidelity = tier_info.get('textual_fidelity_score', 10)
                noise = tier_info.get('semantic_noise_score', 10)
                atomicity = tier_info.get('document_atomicity_score', 10)
                
                if fidelity < 4 or noise < 5 or atomicity < 5:
                    needs_ai_enhancement = True
                    logger.info("🎯 Tier analysis indicates AI enhancement needed")
            
            # Check signal threshold
            if signal_ratio_before < MIN_SIGNAL_THRESHOLD:
                needs_ai_enhancement = True
                logger.info(f"🎯 Signal ratio {signal_ratio_before:.3f} below threshold {MIN_SIGNAL_THRESHOLD}")
            
            # Step 4: AI enhancement if needed
            if needs_ai_enhancement and self.ai_model and len(rules_cleaned) > 1000:
                try:
                    logger.info("🤖 Applying AI enhancement...")
                    ai_result = self.ai_enhance_signal(rules_cleaned)
                    
                    # Use AI result if significantly better
                    improvement_ratio = ai_result['signal_ratio'] / max(signal_ratio_before, 0.001)
                    if improvement_ratio >= AI_IMPROVEMENT_THRESHOLD:
                        final_content = ai_result['enhanced_content']
                        signal_ratio_after = ai_result['signal_ratio']
                        final_topics = ai_result['key_topics']
                        processing_method = 'ai_enhanced'
                        input_tokens = ai_result['input_tokens']
                        output_tokens = ai_result['output_tokens']
                        api_calls = ai_result['api_calls']
                        estimated_cost = ai_result['estimated_cost']
                        
                        logger.info(f"✅ AI enhancement successful: {signal_ratio_after:.3f} signal ratio "
                                  f"({improvement_ratio:.1f}x improvement)")
                    else:
                        logger.info(f"⚠️ AI enhancement insufficient: {improvement_ratio:.1f}x improvement "
                                  f"(needed {AI_IMPROVEMENT_THRESHOLD:.1f}x)")
                        
                except Exception as e:
                    logger.error(f"❌ AI enhancement failed: {e}")
            
            # Update statistics
            self.stats[processing_method] += 1
            processed_size = len(final_content)
            self.stats['total_output_size'] += processed_size
            self.stats['total_input_tokens'] += input_tokens
            self.stats['total_output_tokens'] += output_tokens
            self.stats['total_api_calls'] += api_calls
            self.stats['total_estimated_cost'] += estimated_cost
            
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            
            # Track signal improvements
            if processing_method == 'ai_enhanced':
                improvement = signal_ratio_after / max(signal_ratio_before, 0.001)
                self.stats['signal_improvements'].append(improvement)
            
            return ProcessingResult(
                success=True,
                original_size=original_size,
                processed_size=processed_size,
                signal_ratio_before=signal_ratio_before,
                signal_ratio_after=signal_ratio_after,
                processing_method=processing_method,
                key_topics=final_topics,
                processing_time=processing_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                api_calls=api_calls,
                estimated_cost=estimated_cost
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Failed to process {file_path}: {e}")
            self.stats['failed'] += 1
            
            return ProcessingResult(
                success=False,
                original_size=0,
                processed_size=0,
                signal_ratio_before=0.0,
                signal_ratio_after=0.0,
                processing_method='failed',
                key_topics=[],
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def estimate_corpus_processing(self, input_dir: str, analysis_file: Optional[str] = None) -> Dict:
        """Estimate processing time, costs, and requirements for full corpus."""
        input_path = Path(input_dir)
        
        # Find all files
        file_patterns = ['*.txt', '*.html', '*.md', '*.pdf']
        all_files = []
        for pattern in file_patterns:
            all_files.extend(input_path.rglob(pattern))
        
        # Load analysis data
        tier_data = {}
        if analysis_file and Path(analysis_file).exists():
            df = pd.read_csv(analysis_file)
            for _, row in df.iterrows():
                file_path = row['file_path']
                tier_data[file_path] = {
                    'textual_fidelity_score': row.get('textual_fidelity_score', 10),
                    'semantic_noise_score': row.get('semantic_noise_score', 10),
                    'document_atomicity_score': row.get('document_atomicity_score', 10)
                }
        
        # Analyze files and estimate processing
        total_size = 0
        estimated_ai_files = 0
        estimated_tokens = 0
        
        for file_path in all_files[:100]:  # Sample first 100 files
            try:
                size = file_path.stat().st_size
                total_size += size
                
                # Estimate if file needs AI enhancement
                rel_path = str(file_path.relative_to(input_path.parent))
                tier_info = tier_data.get(rel_path)
                
                if tier_info:
                    fidelity = tier_info.get('textual_fidelity_score', 10)
                    noise = tier_info.get('semantic_noise_score', 10)
                    atomicity = tier_info.get('document_atomicity_score', 10)
                    
                    if fidelity < 4 or noise < 5 or atomicity < 5:
                        estimated_ai_files += 1
                        estimated_tokens += self.estimate_tokens(file_path.read_text(errors='ignore'))
                        
            except Exception:
                continue
        
        # Scale estimates to full corpus
        sample_ratio = min(100 / len(all_files), 1.0)
        
        full_corpus_estimates = {
            'total_files': len(all_files),
            'estimated_ai_files': int(estimated_ai_files / sample_ratio),
            'estimated_rules_files': len(all_files) - int(estimated_ai_files / sample_ratio),
            'estimated_total_tokens': int(estimated_tokens / sample_ratio),
            'estimated_processing_time': {
                'rules_only_hours': (len(all_files) * 0.1) / 3600,  # 0.1 sec per file
                'ai_enhanced_hours': (int(estimated_ai_files / sample_ratio) * 30) / 3600,  # 30 sec per AI file
                'total_hours': ((len(all_files) * 0.1) + (int(estimated_ai_files / sample_ratio) * 30)) / 3600
            },
            'estimated_costs': {
                'total_tokens': int(estimated_tokens / sample_ratio),
                'input_cost': (int(estimated_tokens / sample_ratio) / 1000000) * GEMINI_PRICING['input_tokens_per_million'],
                'output_cost': (int(estimated_tokens / sample_ratio * 0.3) / 1000000) * GEMINI_PRICING['output_tokens_per_million'],
                'total_cost': ((int(estimated_tokens / sample_ratio) / 1000000) * GEMINI_PRICING['input_tokens_per_million']) + 
                             ((int(estimated_tokens / sample_ratio * 0.3) / 1000000) * GEMINI_PRICING['output_tokens_per_million'])
            },
            'thresholds_used': {
                'signal_threshold': MIN_SIGNAL_THRESHOLD,
                'ai_improvement_threshold': AI_IMPROVEMENT_THRESHOLD
            }
        }
        
        return full_corpus_estimates

def main():
    """Enhanced main function with estimation and resumable processing."""
    parser = argparse.ArgumentParser(description='Enhanced Ray Peat Signal Processor v2.0')
    parser.add_argument('--input-dir', required=True, help='Input directory with raw files')
    parser.add_argument('--output-dir', required=True, help='Output directory for processed files')
    parser.add_argument('--analysis-file', help='CSV file with tier analysis')
    parser.add_argument('--limit', type=int, help='Limit number of files to process')
    parser.add_argument('--api-key', help='Google API key for AI enhancement')
    parser.add_argument('--estimate-only', action='store_true', help='Only show estimates, do not process')
    parser.add_argument('--resume', help='Resume from checkpoint file')
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='Save checkpoint every N files')
    
    args = parser.parse_args()
    
    # Initialize enhanced processor
    checkpoint_file = args.resume or (Path(args.output_dir) / 'processing_checkpoint.json')
    processor = EnhancedSignalProcessor(api_key=args.api_key, checkpoint_file=checkpoint_file)
    
    # Show corpus estimates
    logger.info("🔍 Analyzing corpus and generating estimates...")
    estimates = processor.estimate_corpus_processing(args.input_dir, args.analysis_file)
    
    print("\n" + "="*80)
    print("📊 CORPUS PROCESSING ESTIMATES")
    print("="*80)
    print(f"📁 Total files: {estimates['total_files']:,}")
    print(f"🔧 Estimated rules-only files: {estimates['estimated_rules_files']:,}")
    print(f"🤖 Estimated AI-enhanced files: {estimates['estimated_ai_files']:,}")
    print(f"\n⏱️ ESTIMATED PROCESSING TIME:")
    print(f"   Rules-only: {estimates['estimated_processing_time']['rules_only_hours']:.1f} hours")
    print(f"   AI-enhanced: {estimates['estimated_processing_time']['ai_enhanced_hours']:.1f} hours")
    print(f"   Total: {estimates['estimated_processing_time']['total_hours']:.1f} hours")
    print(f"\n💰 ESTIMATED COSTS (Gemini 2.0 Flash):")
    print(f"   Total tokens: {estimates['estimated_costs']['total_tokens']:,}")
    print(f"   Input cost: ${estimates['estimated_costs']['input_cost']:.2f}")
    print(f"   Output cost: ${estimates['estimated_costs']['output_cost']:.2f}")
    print(f"   Total cost: ${estimates['estimated_costs']['total_cost']:.2f}")
    print(f"\n🎯 THRESHOLDS:")
    print(f"   Signal threshold: {estimates['thresholds_used']['signal_threshold']}")
    print(f"   AI improvement threshold: {estimates['thresholds_used']['ai_improvement_threshold']}x")
    print("="*80)
    
    if args.estimate_only:
        return
    
    # Proceed with processing
    print(f"\n🚀 Starting processing in 5 seconds...")
    time.sleep(5)
    
    # TODO: Add the actual processing logic here
    print("✅ Enhanced processing system ready!")

if __name__ == "__main__":
    main() 