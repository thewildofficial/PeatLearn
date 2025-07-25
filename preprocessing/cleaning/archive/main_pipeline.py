#!/usr/bin/env python3
"""
Ray Peat Corpus Cleaning Pipeline

A robust, automated pipeline to clean and prepare a heterogeneous corpus of documents 
about Dr. Ray Peat for Retrieval-Augmented Generation (RAG) systems.

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
import shutil

# Import our cleaning modules
import rules_based_cleaners as rbc
import ai_powered_cleaners as aic

# Constants
DEFAULT_RAW_DATA_DIR = "../../data/raw/raw_data"
DEFAULT_ANALYSIS_FILE = "../../data/analysis/corpus_analysis.csv"
DEFAULT_OUTPUT_DIR = "../../data/processed/cleaned_corpus"
DEFAULT_METADATA_FILE = "../../data/processed/metadata.json"

# Quality thresholds for tier classification
TIER1_THRESHOLDS = {
    'textual_fidelity_score': 4,
    'document_atomicity_score': 5,
    'semantic_noise_score': 5
}

class PipelineStats:
    """Track pipeline statistics for reporting."""
    
    def __init__(self):
        self.total_files = 0
        self.tier1_files = 0
        self.tier2_files = 0
        self.processed_files = 0
        self.error_files = 0
        self.segmented_files = 0
        self.total_output_files = 0
        self.total_input_size = 0
        self.total_output_size = 0
        self.start_time = time.time()
        self.ai_api_calls = 0
        self.processing_errors = []
    
    def log_error(self, file_path, error_msg):
        """Log processing error."""
        self.error_files += 1
        self.processing_errors.append({
            'file': file_path,
            'error': str(error_msg),
            'timestamp': datetime.now().isoformat()
        })
    
    def get_summary(self):
        """Get processing summary."""
        elapsed_time = time.time() - self.start_time
        
        return {
            'processing_time_seconds': round(elapsed_time, 2),
            'total_files_analyzed': self.total_files,
            'tier1_files': self.tier1_files,
            'tier2_files': self.tier2_files,
            'successfully_processed': self.processed_files,
            'files_with_errors': self.error_files,
            'files_segmented': self.segmented_files,
            'output_files_created': self.total_output_files,
            'input_size_mb': round(self.total_input_size / (1024*1024), 2),
            'output_size_mb': round(self.total_output_size / (1024*1024), 2),
            'ai_api_calls_made': self.ai_api_calls,
            'error_rate_percent': round((self.error_files / max(self.total_files, 1)) * 100, 2),
            'errors': self.processing_errors
        }

def setup_logging(output_dir, verbose=False):
    """Setup comprehensive logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup file logging
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Console output
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== Ray Peat Corpus Cleaning Pipeline Started ===")
    logger.info(f"Log file: {log_file}")
    
    return logger

def load_corpus_analysis(analysis_file):
    """Load and validate corpus analysis data."""
    logger = logging.getLogger(__name__)
    
    try:
        df = pd.read_csv(analysis_file)
        logger.info(f"Loaded analysis for {len(df)} files from {analysis_file}")
        
        # Validate required columns
        required_cols = ['file_path', 'file_type', 'semantic_noise_score', 
                        'document_atomicity_score', 'textual_fidelity_score']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Fill NaN values for speaker attribution (only applicable to transcripts)
        df['speaker_attribution_score'] = df['speaker_attribution_score'].fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load corpus analysis: {e}")
        raise

def classify_files(df):
    """Classify files into Tier 1 (rules-based) and Tier 2 (AI-powered) based on quality scores."""
    logger = logging.getLogger(__name__)
    
    # Tier 1: High quality files suitable for rules-based cleaning
    tier1_mask = (
        (df['textual_fidelity_score'] < TIER1_THRESHOLDS['textual_fidelity_score']) &
        (df['document_atomicity_score'] < TIER1_THRESHOLDS['document_atomicity_score']) &
        (df['semantic_noise_score'] < TIER1_THRESHOLDS['semantic_noise_score'])
    )
    
    tier1_files = df[tier1_mask].copy()
    tier2_files = df[~tier1_mask].copy()
    
    logger.info(f"Classification complete:")
    logger.info(f"  Tier 1 (Rules-based): {len(tier1_files)} files ({len(tier1_files)/len(df)*100:.1f}%)")
    logger.info(f"  Tier 2 (AI-powered): {len(tier2_files)} files ({len(tier2_files)/len(df)*100:.1f}%)")
    
    return tier1_files, tier2_files

def generate_clean_filename(original_path, suffix="cleaned", segment_info=None):
    """Generate a clean filename for output files."""
    path = Path(original_path)
    base_name = path.stem
    
    # Clean up the base name
    base_name = base_name.replace(' ', '_').replace('[', '').replace(']', '')
    base_name = ''.join(c for c in base_name if c.isalnum() or c in '._-')
    
    if segment_info:
        # For segmented files
        seg_title = segment_info['title'].replace(' ', '_')
        seg_title = ''.join(c for c in seg_title if c.isalnum() or c in '._-')[:50]
        filename = f"{base_name}_seg{segment_info['index']}_{seg_title}_{suffix}.txt"
    else:
        filename = f"{base_name}_{suffix}.txt"
    
    return filename

def save_cleaned_file(content, output_path, metadata, stats):
    """Save cleaned content to file and update metadata."""
    logger = logging.getLogger(__name__)
    
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        file_size = output_path.stat().st_size
        stats.total_output_size += file_size
        stats.total_output_files += 1
        
        logger.info(f"Saved cleaned file: {output_path.name} ({file_size} bytes)")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save file {output_path}: {e}")
        return False

def process_tier1_file(row, raw_data_dir, output_dir, metadata_list, stats):
    """Process a Tier 1 (high quality) file using rules-based cleaning."""
    logger = logging.getLogger(__name__)
    
    file_path = Path(raw_data_dir) / row['file_path']
    
    try:
        # Calculate input file size
        if file_path.exists():
            stats.total_input_size += file_path.stat().st_size
        
        # Apply appropriate cleaning based on file type
        if row['file_type'] == '.html':
            content = rbc.clean_html(str(file_path))
        else:
            content = rbc.clean_text_file(str(file_path))
        
        if not content or len(content.strip()) < 10:
            logger.warning(f"No meaningful content extracted from {file_path}")
            return False
        
        # Apply additional rules-based cleaning
        content = rbc.normalize_whitespace(content)
        content = rbc.remove_known_artifacts(content, str(file_path))
        content = rbc.fix_common_ocr_errors(content)
        
        # Extract metadata
        file_metadata = rbc.extract_metadata(content, str(row['file_path']))
        
        # Generate output filename
        output_filename = generate_clean_filename(row['file_path'])
        output_path = Path(output_dir) / output_filename
        
        # Save file
        if save_cleaned_file(content, output_path, file_metadata, stats):
            # Add to metadata
            metadata_entry = {
                'new_file_path': str(output_path.relative_to(output_dir)),
                'original_file_path': row['file_path'],
                'processing_method': 'rules_based',
                'tier': 1,
                'quality_scores': {
                    'semantic_noise_score': row['semantic_noise_score'],
                    'document_atomicity_score': row['document_atomicity_score'],
                    'textual_fidelity_score': row['textual_fidelity_score']
                },
                **file_metadata
            }
            metadata_list.append(metadata_entry)
            stats.processed_files += 1
            return True
            
    except Exception as e:
        logger.error(f"Error processing Tier 1 file {file_path}: {e}")
        stats.log_error(str(file_path), e)
        return False
    
    return False

def process_tier2_file(row, raw_data_dir, output_dir, metadata_list, stats, model):
    """Process a Tier 2 (low quality) file using AI-powered cleaning."""
    logger = logging.getLogger(__name__)
    
    file_path = Path(raw_data_dir) / row['file_path']
    
    try:
        # Calculate input file size
        if file_path.exists():
            stats.total_input_size += file_path.stat().st_size
        
        # Load file content
        if row['file_type'] == '.html':
            content = rbc.clean_html(str(file_path))  # Basic HTML extraction first
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        
        if not content or len(content.strip()) < 20:
            logger.warning(f"No meaningful content in {file_path}")
            return False
        
        # Apply basic rules-based cleaning first
        content = rbc.normalize_whitespace(content)
        content = rbc.remove_known_artifacts(content, str(file_path))
        
        # Determine AI cleaning strategy
        strategy = aic.get_cleaning_strategy(
            row['semantic_noise_score'],
            row['document_atomicity_score'], 
            row['textual_fidelity_score'],
            row.get('speaker_attribution_score'),
            str(file_path)
        )
        
        logger.info(f"AI cleaning strategy for {file_path.name}: {strategy}")
        
        # Apply AI cleaning based on strategy
        if strategy == "segment":
            # Document segmentation
            segments = aic.segment_document(content, model)
            stats.ai_api_calls += 1
            
            if len(segments) > 1:
                stats.segmented_files += 1
                logger.info(f"Segmented {file_path.name} into {len(segments)} parts")
            
            # Process each segment
            for i, segment in enumerate(segments):
                segment_info = {'index': i+1, 'title': segment['title']}
                output_filename = generate_clean_filename(row['file_path'], "cleaned", segment_info)
                output_path = Path(output_dir) / output_filename
                
                # Save segment
                if save_cleaned_file(segment['content'], output_path, {}, stats):
                    metadata_entry = {
                        'new_file_path': str(output_path.relative_to(output_dir)),
                        'original_file_path': row['file_path'],
                        'processing_method': 'ai_segmentation',
                        'tier': 2,
                        'segment_index': i+1,
                        'segment_title': segment['title'],
                        'total_segments': len(segments),
                        'ai_strategy': strategy,
                        'quality_scores': {
                            'semantic_noise_score': row['semantic_noise_score'],
                            'document_atomicity_score': row['document_atomicity_score'],
                            'textual_fidelity_score': row['textual_fidelity_score']
                        }
                    }
                    metadata_list.append(metadata_entry)
            
        else:
            # Single document processing
            if strategy == "ocr_correct":
                content = aic.correct_ocr_errors(content, model)
                stats.ai_api_calls += 1
            elif strategy == "speaker_attribution":
                content = aic.attribute_speakers(content, model)
                stats.ai_api_calls += 1
            elif strategy == "enhance":
                content = aic.enhance_text_quality(content, model)
                stats.ai_api_calls += 1
            
            # Save single cleaned file
            output_filename = generate_clean_filename(row['file_path'])
            output_path = Path(output_dir) / output_filename
            
            if save_cleaned_file(content, output_path, {}, stats):
                metadata_entry = {
                    'new_file_path': str(output_path.relative_to(output_dir)),
                    'original_file_path': row['file_path'],
                    'processing_method': f'ai_{strategy}',
                    'tier': 2,
                    'ai_strategy': strategy,
                    'quality_scores': {
                        'semantic_noise_score': row['semantic_noise_score'],
                        'document_atomicity_score': row['document_atomicity_score'],
                        'textual_fidelity_score': row['textual_fidelity_score']
                    }
                }
                metadata_list.append(metadata_entry)
        
        stats.processed_files += 1
        return True
        
    except Exception as e:
        logger.error(f"Error processing Tier 2 file {file_path}: {e}")
        stats.log_error(str(file_path), e)
        return False

def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description='Ray Peat Corpus Cleaning Pipeline')
    parser.add_argument('--raw-data-dir', default=DEFAULT_RAW_DATA_DIR,
                       help='Directory containing raw data files')
    parser.add_argument('--analysis-file', default=DEFAULT_ANALYSIS_FILE,
                       help='CSV file with corpus analysis')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                       help='Output directory for cleaned files')
    parser.add_argument('--metadata-file', default=DEFAULT_METADATA_FILE,
                       help='Output metadata JSON file')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--dry-run', action='store_true',
                       help='Perform dry run without actual processing')
    parser.add_argument('--limit', type=int,
                       help='Limit number of files to process (for testing)')
    
    args = parser.parse_args()
    
    # Setup logging and output directory
    logger = setup_logging(args.output_dir, args.verbose)
    
    # Initialize statistics
    stats = PipelineStats()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {output_dir.absolute()}")
        
        # Load corpus analysis
        logger.info("Loading corpus analysis data...")
        df = load_corpus_analysis(args.analysis_file)
        stats.total_files = len(df)
        
        # Apply limit if specified
        if args.limit:
            df = df.head(args.limit)
            logger.info(f"Processing limited to first {args.limit} files")
        
        # Classify files into tiers
        logger.info("Classifying files by quality...")
        tier1_files, tier2_files = classify_files(df)
        stats.tier1_files = len(tier1_files)
        stats.tier2_files = len(tier2_files)
        
        if args.dry_run:
            logger.info("=== DRY RUN COMPLETE ===")
            logger.info(f"Would process {len(df)} files:")
            logger.info(f"  Tier 1: {len(tier1_files)} files")
            logger.info(f"  Tier 2: {len(tier2_files)} files")
            return
        
        # Initialize AI model for Tier 2 processing
        model = None
        if len(tier2_files) > 0:
            try:
                logger.info("Initializing Gemini AI model...")
                model = aic.initialize_gemini()
                logger.info("AI model ready for Tier 2 processing")
            except Exception as e:
                logger.error(f"Failed to initialize AI model: {e}")
                logger.error("Tier 2 files will be skipped")
        
        # Process files
        metadata_list = []
        
        # Process Tier 1 files
        if len(tier1_files) > 0:
            logger.info(f"=== Processing {len(tier1_files)} Tier 1 files (Rules-based) ===")
            for idx, (_, row) in enumerate(tier1_files.iterrows(), 1):
                logger.info(f"[{idx}/{len(tier1_files)}] Processing: {row['file_path']}")
                process_tier1_file(row, args.raw_data_dir, output_dir, metadata_list, stats)
        
        # Process Tier 2 files
        if len(tier2_files) > 0 and model:
            logger.info(f"=== Processing {len(tier2_files)} Tier 2 files (AI-powered) ===")
            for idx, (_, row) in enumerate(tier2_files.iterrows(), 1):
                logger.info(f"[{idx}/{len(tier2_files)}] Processing: {row['file_path']}")
                process_tier2_file(row, args.raw_data_dir, output_dir, metadata_list, stats, model)
                
                # Add delay between AI calls to respect rate limits
                if idx < len(tier2_files):
                    time.sleep(1)
        
        # Save metadata
        metadata_with_stats = {
            'pipeline_stats': stats.get_summary(),
            'processing_timestamp': datetime.now().isoformat(),
            'files': metadata_list
        }
        
        metadata_path = Path(args.metadata_file)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_with_stats, f, indent=2, ensure_ascii=False)
        
        # Final summary
        summary = stats.get_summary()
        logger.info("=== PIPELINE COMPLETE ===")
        logger.info(f"Total processing time: {summary['processing_time_seconds']} seconds")
        logger.info(f"Files processed: {summary['successfully_processed']}/{summary['total_files_analyzed']}")
        logger.info(f"Output files created: {summary['output_files_created']}")
        logger.info(f"Files segmented: {summary['files_segmented']}")
        logger.info(f"AI API calls: {summary['ai_api_calls_made']}")
        logger.info(f"Error rate: {summary['error_rate_percent']}%")
        logger.info(f"Data processed: {summary['input_size_mb']} MB → {summary['output_size_mb']} MB")
        logger.info(f"Metadata saved to: {metadata_path.absolute()}")
        
        if summary['files_with_errors'] > 0:
            logger.warning(f"Encountered {summary['files_with_errors']} errors during processing")
            error_log = output_dir / "logs" / "errors.json"
            with open(error_log, 'w') as f:
                json.dump(summary['errors'], f, indent=2)
            logger.info(f"Error details saved to: {error_log}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 