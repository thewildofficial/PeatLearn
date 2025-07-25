#!/usr/bin/env python3
"""
Enhanced Ray Peat Signal Extraction Pipeline

This script re-processes the Tier 1 cleaned files to extract pure Ray Peat signal,
removing ads, introductions, and off-topic content while preserving clear speaker attribution.

With million-token context limit, we can be very thorough in our analysis.
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('signal_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enhanced prompts for Ray Peat signal extraction with guaranteed JSON output
RAY_PEAT_SIGNAL_EXTRACTION_PROMPT = """You are an expert at extracting Ray Peat's bioenergetic wisdom. Extract pure Ray Peat signal from this content.

REMOVE: ads, introductions, show logistics, social talk, website mentions
PRESERVE: Ray Peat's explanations, Q&A, scientific discussions, recommendations

For conversations, use: **RAY PEAT:** [his words] and **HOST:** [relevant questions only]

Return ONLY this JSON (no other text):
{{
  "extracted_content": "The cleaned content with speaker attribution",
  "content_type": "conversation",
  "source_analysis": {{
    "original_length": {original_length},
    "signal_ratio": 50,
    "ray_peat_percentage": 30
  }},
  "bioenergetic_content": {{
    "primary_topics": ["thyroid", "metabolism"],
    "mechanisms_explained": ["mechanism1", "mechanism2"],
    "recommendations": ["rec1", "rec2"]
  }},
  "quality_assessment": {{
    "signal_density": "medium",
    "educational_value": "good"
  }},
  "extraction_notes": "Extraction completed"
}}

Content: {content}
"""

ADVANCED_SEGMENTATION_PROMPT = """
You are an expert at organizing Ray Peat's bioenergetic teachings into coherent, self-contained educational segments.

Take this extracted Ray Peat signal and create focused segments where each one:
- Teaches a specific bioenergetic concept or mechanism
- Contains substantial educational content (minimum 300 words)
- Is self-contained and makes sense independently
- Maintains proper context and scientific accuracy
- Preserves Ray Peat's voice and teaching style

SEGMENTATION PRINCIPLES:
1. Topic Coherence: Each segment focuses on one main concept
2. Educational Completeness: Contains enough detail to be instructive
3. Scientific Accuracy: Preserves the precision of Ray Peat's explanations
4. Practical Value: Includes actionable insights where available
5. Logical Flow: Information is presented in a clear, logical sequence

SEGMENT ENHANCEMENT:
For each segment, add:
- Clear topic title and learning objectives
- Key mechanisms or principles explained
- Practical applications and recommendations
- Related concepts and connections to other bioenergetic principles

OUTPUT FORMAT:
Return a detailed JSON object:
{
  "segments": [
    {
      "segment_id": 1,
      "title": "Clear, descriptive title of the bioenergetic concept",
      "learning_objectives": ["what readers will learn from this segment"],
      "content": "The complete segmented content with speaker attribution",
      "bioenergetic_focus": {
        "primary_concept": "main bioenergetic principle",
        "mechanisms": ["physiological processes explained"],
        "applications": ["practical recommendations"],
        "related_topics": ["connected bioenergetic concepts"]
      },
      "quality_metrics": {
        "word_count": number,
        "ray_peat_content_ratio": percentage,
        "educational_density": "very_high|high|medium|low",
        "practical_value": "excellent|good|fair|minimal"
      },
      "keywords": ["key", "bioenergetic", "terms", "concepts"]
    }
  ],
  "segmentation_summary": {
    "total_segments": number,
    "average_length": number_of_words,
    "topic_coverage": ["all main topics covered"],
    "educational_completeness": "comprehensive|good|partial|incomplete"
  },
  "organization_notes": "Explanation of how content was organized and any important considerations"
}

CLEANED RAY PEAT CONTENT TO SEGMENT:
{content}
"""

class RayPeatSignalExtractor:
    """Enhanced signal extraction specifically for Ray Peat content."""
    
    def __init__(self, api_key: str, model_name: str = 'gemini-2.0-flash-exp'):
        """Initialize with Gemini API."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Initialized Ray Peat Signal Extractor with {model_name}")
    
    def extract_signal(self, content: str, max_tokens: int = 900000) -> dict:
        """Extract pure Ray Peat signal from noisy content."""
        if not content or len(content.strip()) < 100:
            return self._empty_result("Content too short")
        
        try:
            # Use generous token limit for thorough analysis
            truncated_content = content[:max_tokens] if len(content) > max_tokens else content
            original_length = len(content)
            
            prompt = RAY_PEAT_SIGNAL_EXTRACTION_PROMPT.format(
                content=truncated_content[:10000],  # Limit content for prompt
                original_length=original_length
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8192,
                    temperature=0.1  # Low temperature for consistent extraction
                )
            )
            
            logger.debug(f"Raw API response: {response.text[:200]}...")
            
            result = self._parse_json_response(response.text)
            
            # Log extraction quality
            signal_ratio = result.get('source_analysis', {}).get('signal_ratio', 0)
            ray_peat_pct = result.get('source_analysis', {}).get('ray_peat_percentage', 0)
            
            logger.info(f"Signal extraction: {signal_ratio}% signal ratio, {ray_peat_pct}% Ray Peat content")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in signal extraction: {e}")
            logger.error(f"Content preview: {content[:200]}...")
            return self._empty_result(f"Extraction error: {str(e)}")
    
    def segment_content(self, content: str, max_tokens: int = 900000) -> dict:
        """Segment extracted content into focused educational units."""
        if not content or len(content.strip()) < 300:
            return {"segments": [], "total_segments": 0, "organization_notes": "Content too short for segmentation"}
        
        try:
            truncated_content = content[:max_tokens] if len(content) > max_tokens else content
            
            prompt = ADVANCED_SEGMENTATION_PROMPT.format(content=truncated_content)
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8192,
                    temperature=0.1
                )
            )
            
            result = self._parse_json_response(response.text)
            
            segment_count = result.get('segmentation_summary', {}).get('total_segments', 0)
            logger.info(f"Content segmented into {segment_count} focused educational units")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in content segmentation: {e}")
            return {"segments": [], "total_segments": 0, "organization_notes": f"Segmentation error: {str(e)}"}
    
    def _parse_json_response(self, response_text: str) -> dict:
        """Parse JSON from AI response, handling various formats."""
        try:
            # Clean up the response
            clean_text = response_text.strip()
            
            # Handle code blocks
            if '```json' in clean_text:
                clean_text = clean_text.split('```json')[1].split('```')[0]
            elif '```' in clean_text:
                # Try to find JSON in any code block
                parts = clean_text.split('```')
                for part in parts:
                    if part.strip().startswith('{'):
                        clean_text = part
                        break
            
            # Find JSON object boundaries more robustly
            start_idx = clean_text.find('{')
            if start_idx == -1:
                raise ValueError("No JSON object found in response")
            
            # Find the matching closing brace
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(clean_text[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            json_str = clean_text[start_idx:end_idx]
            return json.loads(json_str)
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            # Return minimal structure
            return self._empty_result("JSON parsing failed")
    
    def _empty_result(self, reason: str) -> dict:
        """Return empty result structure."""
        return {
            "extracted_content": "",
            "content_type": "error",
            "source_analysis": {
                "original_length": 0,
                "extracted_length": 0,
                "signal_ratio": 0,
                "ray_peat_percentage": 0
            },
            "bioenergetic_content": {
                "primary_topics": [],
                "mechanisms_explained": [],
                "recommendations": [],
                "research_mentioned": []
            },
            "quality_assessment": {
                "signal_density": "low",
                "educational_value": "poor",
                "uniqueness": "basic_information",
                "completeness": "fragmentary"
            },
            "extraction_notes": reason
        }

def process_tier1_files(input_dir: str, output_dir: str, api_key: str, limit: int = None):
    """
    Process all Tier 1 files with enhanced signal extraction.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    extractor = RayPeatSignalExtractor(api_key)
    
    # Statistics tracking
    stats = {
        "start_time": datetime.now().isoformat(),
        "files_processed": 0,
        "files_enhanced": 0,
        "segments_created": 0,
        "total_signal_extracted": 0,
        "high_quality_extractions": 0,
        "processing_errors": 0
    }
    
    # Get all text files
    txt_files = list(input_path.glob("*.txt"))
    if limit:
        txt_files = txt_files[:limit]
    
    logger.info(f"Processing {len(txt_files)} Tier 1 files for signal enhancement")
    
    for i, file_path in enumerate(txt_files, 1):
        try:
            logger.info(f"[{i}/{len(txt_files)}] Processing: {file_path.name}")
            
            # Read original content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Extract pure Ray Peat signal
            extraction_result = extractor.extract_signal(original_content)
            
            # Check if extraction was successful and valuable
            quality = extraction_result.get('quality_assessment', {})
            signal_density = quality.get('signal_density', 'low')
            ray_peat_pct = extraction_result.get('source_analysis', {}).get('ray_peat_percentage', 0)
            
            if signal_density in ['high', 'very_high', 'medium'] and ray_peat_pct >= 15:
                
                extracted_content = extraction_result['extracted_content']
                
                # Segment the content if it's substantial
                if len(extracted_content) > 1000:
                    segmentation_result = extractor.segment_content(extracted_content)
                    
                    # Save each segment as a separate file
                    for segment in segmentation_result.get('segments', []):
                        segment_filename = f"{file_path.stem}_segment_{segment['segment_id']:02d}_{segment['title'].replace(' ', '_')[:50]}.txt"
                        segment_path = output_path / segment_filename
                        
                        # Create rich content with metadata
                        segment_content = _create_enhanced_segment_content(
                            segment, extraction_result, file_path.name
                        )
                        
                        with open(segment_path, 'w', encoding='utf-8') as f:
                            f.write(segment_content)
                        
                        stats["segments_created"] += 1
                        logger.info(f"  Created segment: {segment['title']}")
                
                else:
                    # Save as single enhanced file
                    enhanced_filename = f"{file_path.stem}_enhanced.txt"
                    enhanced_path = output_path / enhanced_filename
                    
                    enhanced_content = _create_enhanced_file_content(
                        extraction_result, file_path.name
                    )
                    
                    with open(enhanced_path, 'w', encoding='utf-8') as f:
                        f.write(enhanced_content)
                
                stats["files_enhanced"] += 1
                if signal_density == 'very_high':
                    stats["high_quality_extractions"] += 1
                
                stats["total_signal_extracted"] += ray_peat_pct
            
            else:
                logger.info(f"  Skipped {file_path.name}: Low signal quality ({signal_density}, {ray_peat_pct}% Ray Peat)")
            
            stats["files_processed"] += 1
            
            # Brief pause to be respectful to API
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            stats["processing_errors"] += 1
            continue
    
    # Save final statistics
    stats["end_time"] = datetime.now().isoformat()
    stats["average_signal_per_file"] = stats["total_signal_extracted"] / max(stats["files_enhanced"], 1)
    
    with open(output_path / "signal_extraction_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Signal extraction complete: {stats}")
    return stats

def _create_enhanced_segment_content(segment: dict, extraction_result: dict, source_file: str) -> str:
    """Create rich content for a segment with comprehensive metadata."""
    
    bioenergetic_focus = segment.get('bioenergetic_focus', {})
    quality_metrics = segment.get('quality_metrics', {})
    
    content = f"""# {segment.get('title', 'Ray Peat Bioenergetic Insight')}

## Source Information
- **Original File:** {source_file}
- **Segment ID:** {segment.get('segment_id', 1)}
- **Content Type:** {extraction_result.get('content_type', 'unknown')}
- **Signal Quality:** {extraction_result.get('quality_assessment', {}).get('signal_density', 'unknown')}

## Learning Objectives
{chr(10).join(f"- {obj}" for obj in segment.get('learning_objectives', []))}

## Bioenergetic Focus
- **Primary Concept:** {bioenergetic_focus.get('primary_concept', 'General bioenergetics')}
- **Key Mechanisms:** {', '.join(bioenergetic_focus.get('mechanisms', []))}
- **Applications:** {', '.join(bioenergetic_focus.get('applications', []))}
- **Related Topics:** {', '.join(bioenergetic_focus.get('related_topics', []))}

## Quality Metrics
- **Word Count:** {quality_metrics.get('word_count', 0)}
- **Ray Peat Content:** {quality_metrics.get('ray_peat_content_ratio', 0)}%
- **Educational Density:** {quality_metrics.get('educational_density', 'unknown')}
- **Practical Value:** {quality_metrics.get('practical_value', 'unknown')}

## Keywords
{', '.join(segment.get('keywords', []))}

---

## Content

{segment.get('content', '')}

---

## Extraction Notes
{extraction_result.get('extraction_notes', 'No additional notes')}
"""
    
    return content

def _create_enhanced_file_content(extraction_result: dict, source_file: str) -> str:
    """Create rich content for a non-segmented file."""
    
    bioenergetic = extraction_result.get('bioenergetic_content', {})
    quality = extraction_result.get('quality_assessment', {})
    source_analysis = extraction_result.get('source_analysis', {})
    
    content = f"""# Ray Peat Bioenergetic Content

## Source Information
- **Original File:** {source_file}
- **Content Type:** {extraction_result.get('content_type', 'unknown')}
- **Signal Ratio:** {source_analysis.get('signal_ratio', 0)}%
- **Ray Peat Content:** {source_analysis.get('ray_peat_percentage', 0)}%

## Quality Assessment
- **Signal Density:** {quality.get('signal_density', 'unknown')}
- **Educational Value:** {quality.get('educational_value', 'unknown')}
- **Uniqueness:** {quality.get('uniqueness', 'unknown')}
- **Completeness:** {quality.get('completeness', 'unknown')}

## Bioenergetic Content Overview
- **Primary Topics:** {', '.join(bioenergetic.get('primary_topics', []))}
- **Mechanisms Explained:** {', '.join(bioenergetic.get('mechanisms_explained', []))}
- **Research Mentioned:** {', '.join(bioenergetic.get('research_mentioned', []))}

## Recommendations
{chr(10).join(f"- {rec}" for rec in bioenergetic.get('recommendations', []))}

---

## Content

{extraction_result.get('extracted_content', '')}

---

## Extraction Notes
{extraction_result.get('extraction_notes', 'No additional notes')}
"""
    
    return content

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Enhanced Ray Peat Signal Extraction")
    parser.add_argument("--input-dir", required=True, help="Directory with Tier 1 cleaned files")
    parser.add_argument("--output-dir", required=True, help="Directory for enhanced output")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    parser.add_argument("--api-key", help="Gemini API key (or use GEMINI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found. Please provide it via --api-key or environment variable.")
        return
    
    # Validate paths
    if not Path(args.input_dir).exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return
    
    # Run processing
    try:
        stats = process_tier1_files(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            api_key=api_key,
            limit=args.limit
        )
        
        logger.info("Ray Peat signal extraction completed successfully!")
        logger.info(f"Enhanced {stats['files_enhanced']} files into {stats['segments_created']} focused segments")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")

if __name__ == "__main__":
    main() 