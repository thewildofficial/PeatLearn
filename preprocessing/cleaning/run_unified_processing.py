#!/usr/bin/env python3
"""
Ray Peat Signal Extraction - Unified Processing Script

This script runs the complete signal extraction pipeline:
1. Rules-based cleaning for all files
2. AI enhancement for low-signal files (when API available)
3. Mega-chunking optimized for million-token context windows
4. Signal quality assessment and reporting

Usage:
    python run_unified_processing.py --input-dir ../../data/raw/raw_data --output-dir ../../data/processed/signal_extracted --limit 10

Author: Aban Hasan
Date: 2025
"""

import argparse
import logging
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Import our processing modules
from unified_signal_processor import UnifiedSignalProcessor
from mega_chunker import MegaChunker

# Load environment variables (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, that's okay

def setup_logging(output_dir: Path, verbose: bool = False):
    """Setup comprehensive logging."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"unified_processing_{timestamp}.log"
    
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Unified Ray Peat Processing Pipeline Started")
    logger.info(f"Log file: {log_file}")
    
    return logger

def process_and_chunk_corpus(
    input_dir: str,
    output_dir: str,
    analysis_file: str = None,
    limit: int = None,
    api_key: str = None,
    create_mega_chunks: bool = True,
    max_chunk_size: int = 900000
):
    """
    Run the complete processing and chunking pipeline.
    
    Args:
        input_dir: Directory with raw Ray Peat files
        output_dir: Directory for processed output
        analysis_file: Optional CSV with quality analysis
        limit: Optional limit on files to process
        api_key: Optional Google API key for AI enhancement
        create_mega_chunks: Whether to create mega-chunks
        max_chunk_size: Maximum size for chunks
    
    Returns:
        Dict with processing summary
    """
    output_path = Path(output_dir)
    logger = logging.getLogger(__name__)
    
    # Phase 1: Signal Processing
    logger.info("="*60)
    logger.info("PHASE 1: RAY PEAT SIGNAL EXTRACTION")
    logger.info("="*60)
    
    processor = UnifiedSignalProcessor(api_key=api_key)
    
    processing_summary = processor.process_corpus(
        input_dir=input_dir,
        output_dir=output_dir,
        analysis_file=analysis_file,
        limit=limit
    )
    
    logger.info(f"Phase 1 completed: {processing_summary['successful']} files processed")
    
    # Phase 2: Mega-Chunking (if requested)
    chunking_summary = {}
    if create_mega_chunks:
        logger.info("="*60)
        logger.info("PHASE 2: MEGA-CHUNK CREATION")
        logger.info("="*60)
        
        # Create chunking output directory
        chunks_dir = output_path / "mega_chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        # Find all processed files
        processed_files = list(output_path.glob("*_processed.txt"))
        logger.info(f"Found {len(processed_files)} processed files to chunk")
        
        chunker = MegaChunker(max_chunk_size=max_chunk_size)
        total_chunks = 0
        chunk_stats = []
        
        for i, processed_file in enumerate(processed_files, 1):
            logger.info(f"Chunking file {i}/{len(processed_files)}: {processed_file.name}")
            
            try:
                # Read processed content
                with open(processed_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create chunks
                chunks = chunker.create_mega_chunks(content)
                
                if chunks:
                    # Save chunks
                    base_name = processed_file.stem.replace('_processed', '')
                    file_chunks_dir = chunks_dir / base_name
                    chunker.save_chunks(content, chunks, file_chunks_dir, base_name)
                    
                    total_chunks += len(chunks)
                    
                    # Collect stats
                    avg_density = sum(c.ray_peat_density for c in chunks) / len(chunks)
                    chunk_stats.append({
                        'file': processed_file.name,
                        'chunks_created': len(chunks),
                        'avg_density': avg_density,
                        'total_chars': sum(c.size_chars for c in chunks),
                        'total_tokens': sum(c.estimated_tokens for c in chunks)
                    })
                    
                    logger.info(f"Created {len(chunks)} chunks, avg density: {avg_density:.3f}")
                else:
                    logger.warning(f"No chunks created for {processed_file.name}")
                    
            except Exception as e:
                logger.error(f"Failed to chunk {processed_file.name}: {e}")
        
        chunking_summary = {
            'files_chunked': len([s for s in chunk_stats if s['chunks_created'] > 0]),
            'total_chunks_created': total_chunks,
            'avg_chunks_per_file': total_chunks / max(len(chunk_stats), 1),
            'avg_signal_density': sum(s['avg_density'] for s in chunk_stats) / max(len(chunk_stats), 1),
            'total_tokens': sum(s['total_tokens'] for s in chunk_stats),
            'chunk_stats': chunk_stats
        }
        
        # Save chunking metadata
        chunking_metadata_file = chunks_dir / "chunking_summary.json"
        with open(chunking_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(chunking_summary, f, indent=2)
        
        logger.info(f"Phase 2 completed: {total_chunks} mega-chunks created")
    
    # Phase 3: Final Summary
    logger.info("="*60)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*60)
    
    final_summary = {
        'timestamp': datetime.now().isoformat(),
        'input_directory': input_dir,
        'output_directory': output_dir,
        'processing_summary': processing_summary,
        'chunking_summary': chunking_summary,
        'pipeline_config': {
            'analysis_file_used': analysis_file is not None,
            'ai_enhancement_available': api_key is not None,
            'file_limit': limit,
            'mega_chunking_enabled': create_mega_chunks,
            'max_chunk_size': max_chunk_size
        }
    }
    
    # Save final summary
    summary_file = output_path / "pipeline_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2)
    
    # Log key metrics
    logger.info(f"📊 FINAL METRICS:")
    logger.info(f"  ✅ Files processed: {processing_summary.get('successful', 0)}")
    logger.info(f"  🔧 Rules-only: {processing_summary.get('rules_only', 0)}")
    logger.info(f"  🤖 AI-enhanced: {processing_summary.get('ai_enhanced', 0)}")
    logger.info(f"  📈 Avg signal ratio: {processing_summary.get('average_signal_ratio', 0):.3f}")
    
    if create_mega_chunks:
        logger.info(f"  📦 Mega-chunks created: {chunking_summary.get('total_chunks_created', 0)}")
        logger.info(f"  🎯 Avg chunk density: {chunking_summary.get('avg_signal_density', 0):.3f}")
        logger.info(f"  🔤 Total tokens: {chunking_summary.get('total_tokens', 0):,}")
    
    logger.info(f"  💾 Output directory: {output_dir}")
    logger.info(f"📋 Complete summary saved to: {summary_file}")
    
    return final_summary

def main():
    """Main entry point for unified processing pipeline."""
    parser = argparse.ArgumentParser(
        description='Unified Ray Peat Signal Extraction & Mega-Chunking Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 10 files with rules-based cleaning only
  python run_unified_processing.py --input-dir ../../data/raw/raw_data --output-dir ../../data/processed/signal_extracted --limit 10

  # Process all files with AI enhancement (requires API key)
  python run_unified_processing.py --input-dir ../../data/raw/raw_data --output-dir ../../data/processed/signal_extracted --analysis-file ../../data/analysis/corpus_analysis.csv --api-key YOUR_API_KEY

  # Custom chunk size for smaller context windows
  python run_unified_processing.py --input-dir ../../data/raw/raw_data --output-dir ../../data/processed/signal_extracted --max-chunk-size 500000
        """
    )
    
    # Required arguments
    parser.add_argument('--input-dir', required=True,
                       help='Directory containing raw Ray Peat files')
    parser.add_argument('--output-dir', required=True,
                       help='Directory for processed output')
    
    # Optional arguments
    parser.add_argument('--analysis-file',
                       help='CSV file with quality analysis (for smarter tier classification)')
    parser.add_argument('--limit', type=int,
                       help='Limit number of files to process (useful for testing)')
    parser.add_argument('--api-key',
                       help='Google API key for AI enhancement (or set GOOGLE_API_KEY env var)')
    parser.add_argument('--no-chunking', action='store_true',
                       help='Skip mega-chunk creation')
    parser.add_argument('--max-chunk-size', type=int, default=900000,
                       help='Maximum chunk size in characters (default: 900,000)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup output directory and logging
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_path, args.verbose)
    
    # Get API key from argument or environment
    api_key = args.api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.warning("No API key provided - AI enhancement will be disabled")
        logger.warning("Set GOOGLE_API_KEY environment variable or use --api-key argument")
    
    # Log configuration
    logger.info("Pipeline Configuration:")
    logger.info(f"  Input directory: {args.input_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Analysis file: {args.analysis_file or 'None'}")
    logger.info(f"  File limit: {args.limit or 'No limit'}")
    logger.info(f"  AI enhancement: {'Enabled' if api_key else 'Disabled'}")
    logger.info(f"  Mega-chunking: {'Disabled' if args.no_chunking else 'Enabled'}")
    logger.info(f"  Max chunk size: {args.max_chunk_size:,} chars")
    
    try:
        start_time = time.time()
        
        # Run the pipeline
        summary = process_and_chunk_corpus(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            analysis_file=args.analysis_file,
            limit=args.limit,
            api_key=api_key,
            create_mega_chunks=not args.no_chunking,
            max_chunk_size=args.max_chunk_size
        )
        
        processing_time = time.time() - start_time
        logger.info(f"🎉 Pipeline completed successfully in {processing_time:.1f} seconds!")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main()) 