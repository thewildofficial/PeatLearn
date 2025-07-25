#!/usr/bin/env python3
"""
Simple runner script for Ray Peat signal enhancement.
This will process your Tier 1 files and create compendious, signal-rich content.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the current directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from enhance_tier1_signal import process_tier1_files

def main():
    """Run the signal enhancement pipeline."""
    
    # Load environment variables
    load_dotenv()
    
    # Configuration
    TIER1_INPUT_DIR = "../../data/processed/cleaned_corpus_tier1"
    ENHANCED_OUTPUT_DIR = "../../data/processed/ray_peat_signal_enhanced"
    
    # API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables!")
        print("Please add it to your .env file or set it as an environment variable.")
        return
    
    # Validate input directory
    input_path = Path(__file__).parent / TIER1_INPUT_DIR
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_path}")
        return
    
    output_path = Path(__file__).parent / ENHANCED_OUTPUT_DIR
    
    print("🚀 Starting Ray Peat Signal Enhancement Pipeline")
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    print("🎯 Goal: Extract pure Ray Peat bioenergetic signal with clear speaker attribution")
    print()
    
    # Process files (start with a small limit for testing)
    stats = process_tier1_files(
        input_dir=str(input_path),
        output_dir=str(output_path),
        api_key=api_key,
        limit=5  # Start with 5 files for testing
    )
    
    print()
    print("✅ Signal Enhancement Complete!")
    print(f"📊 Files processed: {stats['files_processed']}")
    print(f"📊 Files enhanced: {stats['files_enhanced']}")
    print(f"📊 Segments created: {stats['segments_created']}")
    print(f"📊 High-quality extractions: {stats['high_quality_extractions']}")
    print(f"📊 Average signal per file: {stats.get('average_signal_per_file', 0):.1f}%")
    
    if stats['segments_created'] > 0:
        print()
        print("🎉 Success! Your enhanced Ray Peat signal corpus is ready!")
        print(f"📍 Location: {output_path}")
        print("📋 Each segment contains:")
        print("   • Pure Ray Peat bioenergetic insights")
        print("   • Clear speaker attribution")
        print("   • Comprehensive metadata")
        print("   • Educational objectives")
        print("   • Key concepts and mechanisms")
    else:
        print()
        print("⚠️  No segments were created. This could mean:")
        print("   • Signal quality was too low in the test files")
        print("   • Content was too short for segmentation")
        print("   • API issues occurred")
        print("Check the logs for details.")

if __name__ == "__main__":
    main() 