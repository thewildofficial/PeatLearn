#!/usr/bin/env python3
"""
Ray Peat Corpus Cleaning Pipeline - Tier 1 Only
Processes only high-quality files using rules-based cleaning.
"""

import argparse
import logging
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

# Import our cleaning modules
import rules_based_cleaners as rbc

# Constants
DEFAULT_RAW_DATA_DIR = "../../data/raw/raw_data"
DEFAULT_ANALYSIS_FILE = "../../data/analysis/corpus_analysis.csv"
DEFAULT_OUTPUT_DIR = "../../data/processed/cleaned_corpus_tier1"
DEFAULT_METADATA_FILE = "../../data/processed/metadata_tier1.json"

# Quality thresholds for tier classification
TIER1_THRESHOLDS = {
    'textual_fidelity_score': 4,
    'document_atomicity_score': 5,
    'semantic_noise_score': 5
}

class PipelineStats:
    """Track pipeline processing statistics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.total_files = 0
        self.tier1_files = 0
        self.tier2_files = 0
        self.processed_files = 0
        self.failed_files = 0
        self.output_files_created = 0
        self.files_segmented = 0
        self.input_size_bytes = 0
        self.output_size_bytes = 0
        
    def get_summary(self):
        """Get summary statistics."""
        processing_time = time.time() - self.start_time
        return {
            'processing_time_seconds': round(processing_time, 1),
            'total_files_analyzed': self.total_files,
            'tier1_files': self.tier1_files,
            'tier2_files': self.tier2_files,
            'successfully_processed': self.processed_files,
            'files_with_errors': self.failed_files,
            'files_segmented': self.files_segmented,
            'output_files_created': self.output_files_created,
            'input_size_mb': round(self.input_size_bytes / (1024*1024), 2),
            'output_size_mb': round(self.output_size_bytes / (1024*1024), 2),
            'error_rate_percent': round((self.failed_files / max(self.processed_files, 1)) * 100, 1)
        }

def setup_logging(output_dir, verbose=False):
    """Setup logging configuration."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_tier1_{timestamp}.log"
    
    level = logging.INFO if not verbose else logging.DEBUG
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(levelname)s:%(name)s:%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    
    return logger

def load_corpus_analysis(analysis_file):
    """Load and validate corpus analysis data."""
    try:
        df = pd.read_csv(analysis_file)
        
        # Validate required columns
        required_cols = ['file_path', 'textual_fidelity_score', 
                        'document_atomicity_score', 'semantic_noise_score']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Fill NaN values with high scores (worst case)
        score_columns = ['textual_fidelity_score', 'document_atomicity_score', 'semantic_noise_score']
        for col in score_columns:
            df[col] = df[col].fillna(10)
        
        return df
        
    except Exception as e:
        logging.error(f"Failed to load corpus analysis: {e}")
        raise

def classify_files(df):
    """Classify files into Tier 1 based on quality scores."""
    
    # Tier 1: High quality files suitable for rules-based cleaning
    tier1_mask = (
        (df['textual_fidelity_score'] < TIER1_THRESHOLDS['textual_fidelity_score']) &
        (df['document_atomicity_score'] < TIER1_THRESHOLDS['document_atomicity_score']) &
        (df['semantic_noise_score'] < TIER1_THRESHOLDS['semantic_noise_score'])
    )
    
    tier1_files = df[tier1_mask]
    tier2_files = df[~tier1_mask]  # For reference only
    
    return tier1_files, tier2_files

def generate_clean_filename(original_path, suffix="cleaned", segment_info=None):
    """Generate standardized filename for cleaned output."""
    path = Path(original_path)
    base_name = path.stem
    
    if segment_info:
        # For segmented files: {original_name}_seg{i}_{title}_{suffix}.txt
        title = segment_info.get('title', 'Document').replace(' ', '_')
        # Clean title for filename
        title = ''.join(c for c in title if c.isalnum() or c in ['_', '-'])[:50]
        clean_name = f"{base_name}_seg{segment_info['index']}_{title}_{suffix}.txt"
    else:
        # For single files: {original_name}_{suffix}.txt
        clean_name = f"{base_name}_{suffix}.txt"
    
    return clean_name

def save_cleaned_file(content, output_path, metadata, stats):
    """Save cleaned content to file and update statistics."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        file_size = output_path.stat().st_size
        stats.output_size_bytes += file_size
        stats.output_files_created += 1
        
        # Update metadata with file info
        metadata.update({
            'file_size_bytes': file_size,
            'word_count': len(content.split()),
            'character_count': len(content)
        })
        
        logging.info(f"Saved cleaned file: {output_path.name} ({file_size} bytes)")
        return True
        
    except Exception as e:
        logging.error(f"Failed to save file {output_path}: {e}")
        return False

def process_tier1_file(row, raw_data_dir, output_dir, metadata_list, stats):
    """Process a single Tier 1 file using rules-based cleaning."""
    
    file_path = row['file_path']
    full_path = Path(raw_data_dir) / file_path
    
    # Check if file exists
    if not full_path.exists():
        logging.warning(f"File not found: {full_path}")
        stats.failed_files += 1
        return
    
    try:
        # Track input file size
        stats.input_size_bytes += full_path.stat().st_size
        
        # Determine file type and apply appropriate cleaning
        if file_path.lower().endswith('.html'):
            # HTML files: extract content and clean
            cleaned_content = rbc.clean_html(full_path)
            if not cleaned_content:
                logging.warning(f"No content extracted from HTML: {file_path}")
                stats.failed_files += 1
                return
        else:
            # Text files: clean directly
            cleaned_content = rbc.clean_text_file(full_path)
            if not cleaned_content:
                logging.warning(f"No content after cleaning: {file_path}")
                stats.failed_files += 1
                return
        
        # Apply common text cleaning
        cleaned_content = rbc.normalize_whitespace(cleaned_content)
        cleaned_content = rbc.remove_known_artifacts(cleaned_content, file_path)
        cleaned_content = rbc.fix_common_ocr_errors(cleaned_content)
        
        # Extract basic metadata
        extracted_metadata = rbc.extract_metadata(cleaned_content, file_path)
        
        # Generate output filename
        clean_filename = generate_clean_filename(file_path)
        output_path = Path(output_dir) / clean_filename
        
        # Create metadata entry
        metadata = {
            'new_file_path': clean_filename,
            'original_file_path': file_path,
            'processing_method': 'rules_based',
            'tier': 1,
            'quality_scores': {
                'textual_fidelity_score': row.get('textual_fidelity_score'),
                'document_atomicity_score': row.get('document_atomicity_score'),
                'semantic_noise_score': row.get('semantic_noise_score'),
                'speaker_attribution_score': row.get('speaker_attribution_score')
            },
            'extracted_title': extracted_metadata.get('title'),
            'extracted_date': extracted_metadata.get('date'),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Save the cleaned file
        if save_cleaned_file(cleaned_content, output_path, metadata, stats):
            metadata_list.append(metadata)
            stats.processed_files += 1
        else:
            stats.failed_files += 1
            
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        stats.failed_files += 1

def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description='Ray Peat Corpus Cleaning Pipeline - Tier 1 Only')
    parser.add_argument('--raw-data-dir', default=DEFAULT_RAW_DATA_DIR,
                       help='Directory containing raw corpus files')
    parser.add_argument('--analysis-file', default=DEFAULT_ANALYSIS_FILE,
                       help='CSV file with corpus analysis and quality scores')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                       help='Directory for cleaned output files')
    parser.add_argument('--metadata-file', default=DEFAULT_METADATA_FILE,
                       help='JSON file for processing metadata')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview processing without actual file changes')
    parser.add_argument('--limit', type=int,
                       help='Limit processing to first N files (for testing)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir, args.verbose)
    
    logger.info("=== Ray Peat Corpus Cleaning Pipeline (Tier 1 Only) Started ===")
    logger.info(f"Output directory: {Path(args.output_dir).resolve()}")
    
    # Initialize statistics
    stats = PipelineStats()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load corpus analysis
        logger.info("Loading corpus analysis data...")
        df = load_corpus_analysis(args.analysis_file)
        stats.total_files = len(df)
        logger.info(f"Loaded analysis for {len(df)} files from {args.analysis_file}")
        
        # Apply limit if specified
        if args.limit:
            df = df.head(args.limit)
            logger.info(f"Processing limited to first {args.limit} files")
        
        # Classify files
        logger.info("Classifying files by quality...")
        tier1_files, tier2_files = classify_files(df)
        stats.tier1_files = len(tier1_files)
        stats.tier2_files = len(tier2_files)
        
        logger.info("Classification complete:")
        logger.info(f"  Tier 1 (Rules-based): {len(tier1_files)} files ({len(tier1_files)/len(df)*100:.1f}%)")
        logger.info(f"  Tier 2 (AI-powered): {len(tier2_files)} files ({len(tier2_files)/len(df)*100:.1f}%)")
        
        # Dry run mode
        if args.dry_run:
            logger.info("=== DRY RUN COMPLETE ===")
            logger.info(f"Would process {len(tier1_files)} Tier 1 files")
            return
        
        # Process Tier 1 files
        metadata_list = []
        
        if len(tier1_files) > 0:
            logger.info(f"=== Processing {len(tier1_files)} Tier 1 files (Rules-based) ===")
            
            for idx, (_, row) in enumerate(tier1_files.iterrows(), 1):
                logger.info(f"[{idx}/{len(tier1_files)}] Processing: {row['file_path']}")
                process_tier1_file(row, args.raw_data_dir, output_dir, metadata_list, stats)
        
        # Save metadata
        metadata_with_stats = {
            'pipeline_stats': stats.get_summary(),
            'processing_timestamp': datetime.now().isoformat(),
            'pipeline_mode': 'tier1_only',
            'files': metadata_list
        }
        
        metadata_path = Path(args.metadata_file)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_with_stats, f, indent=2, ensure_ascii=False)
        
        # Final summary
        summary = stats.get_summary()
        logger.info("=== PIPELINE COMPLETE ===")
        logger.info(f"Total processing time: {summary['processing_time_seconds']} seconds")
        logger.info(f"Files processed: {summary['successfully_processed']}/{stats.total_files}")
        logger.info(f"Output files created: {summary['output_files_created']}")
        logger.info(f"Error rate: {summary['error_rate_percent']}%")
        logger.info(f"Data processed: {summary['input_size_mb']} MB → {summary['output_size_mb']} MB")
        logger.info(f"Metadata saved to: {metadata_path.resolve()}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 