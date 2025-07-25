#!/usr/bin/env python3
"""
Enhanced Ray Peat Signal Extraction

Handles large documents by intelligent chunking while preserving context.
Removes truncation issues and supports very large newsletters/books.
"""

import os
import re
import logging
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced prompt for larger context window
EXTRACT_RAY_PEAT_PROMPT = """You are extracting Ray Peat's bioenergetic wisdom from this content.

REMOVE COMPLETELY:
- All commercial advertisements and sponsorships  
- Host introductions and show logistics
- Social talk and off-topic conversations
- Website mentions and contact information
- Caller management and technical issues

KEEP AND CLEAN:
- All Ray Peat explanations and insights
- Relevant host questions that provide context
- Scientific discussions and recommendations
- Q&A exchanges about health topics

Format conversations as:
**RAY PEAT:** [his exact words, cleaned]
**HOST:** [only relevant questions/context]

Extract ALL valuable content - do not truncate or summarize.

Content to clean:
{content}
"""

class EnhancedSignalExtractor:
    """Enhanced AI-powered signal extraction with large document support."""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info("Enhanced signal extractor initialized")
    
    def chunk_content(self, content: str, max_chunk_size: int = 800000) -> List[Tuple[str, int, int]]:
        """Intelligently chunk content preserving conversation boundaries."""
        if len(content) <= max_chunk_size:
            return [(content, 0, len(content))]
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(content):
            # Calculate chunk end position
            chunk_end = min(current_pos + max_chunk_size, len(content))
            
            # If not at the end, try to find a good break point
            if chunk_end < len(content):
                # Look for paragraph breaks or speaker changes within last 10%
                search_start = max(current_pos, chunk_end - max_chunk_size // 10)
                
                # Look for natural break points
                break_patterns = [
                    r'\n\n\*\*RAY PEAT:\*\*',  # Speaker change
                    r'\n\n\*\*HOST:\*\*',     # Speaker change
                    r'\n\n[A-Z]',              # New paragraph
                    r'\.\s*\n\n',              # Sentence + paragraph break
                ]
                
                best_break = chunk_end
                for pattern in break_patterns:
                    matches = list(re.finditer(pattern, content[search_start:chunk_end]))
                    if matches:
                        # Use the last match to avoid cutting mid-sentence
                        best_break = search_start + matches[-1].start()
                        break
                
                chunk_end = best_break
            
            chunk_text = content[current_pos:chunk_end]
            chunks.append((chunk_text, current_pos, chunk_end))
            current_pos = chunk_end
            
            logger.info(f"Created chunk: {len(chunk_text)} chars (pos {current_pos}-{chunk_end})")
        
        return chunks
    
    def extract_signal(self, content: str, preserve_all: bool = True) -> str:
        """Extract Ray Peat signal from content with full preservation."""
        if not content or len(content.strip()) < 200:
            return ""
        
        try:
            # Handle large content by chunking
            chunks = self.chunk_content(content)
            extracted_parts = []
            
            for i, (chunk, start_pos, end_pos) in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                
                prompt = EXTRACT_RAY_PEAT_PROMPT.format(content=chunk)
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=8192,  # Much higher limit
                        temperature=0.1
                    )
                )
                
                chunk_result = response.text.strip()
                
                # Quality check
                if len(chunk_result) < 100:
                    logger.warning(f"Chunk {i+1} produced very short result")
                    continue
                
                # Check for Ray Peat content
                ray_peat_indicators = ['ray peat', 'thyroid', 'progesterone', 'pufa', 'metabolism', 'bioenergetic']
                if not any(indicator in chunk_result.lower() for indicator in ray_peat_indicators):
                    logger.warning(f"Chunk {i+1} may not contain Ray Peat content")
                
                extracted_parts.append(chunk_result)
            
            # Combine all parts
            if not extracted_parts:
                logger.error("No valid content extracted from any chunk")
                return ""
            
            # Join chunks with clear separation
            combined_result = "\n\n---\n\n".join(extracted_parts)
            
            # Final cleanup - remove duplicate separators
            combined_result = re.sub(r'\n---\n---\n', '\n---\n', combined_result)
            
            logger.info(f"Successfully extracted {len(combined_result)} characters from {len(chunks)} chunks")
            return combined_result
            
        except Exception as e:
            logger.error(f"Signal extraction failed: {e}")
            return ""

def process_file(input_path: Path, output_path: Path, extractor: EnhancedSignalExtractor) -> bool:
    """Process a single file with enhanced extraction."""
    try:
        # Read input
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_length = len(content)
        logger.info(f"Processing {input_path.name} ({original_length:,} chars)")
        
        # Extract signal
        extracted_content = extractor.extract_signal(content)
        
        if not extracted_content:
            logger.warning(f"No signal extracted from {input_path.name}")
            return False
        
        # Calculate metrics
        extracted_length = len(extracted_content)
        noise_reduction = ((original_length - extracted_length) / original_length) * 100
        
        # Create output with metadata
        output_content = f"""# Ray Peat Content - {input_path.stem}

**Source:** {input_path.name}
**Original Length:** {original_length:,} characters
**Cleaned Length:** {extracted_length:,} characters  
**Noise Reduction:** {noise_reduction:.1f}%

---

{extracted_content}

---

*Extracted using AI signal enhancement to preserve Ray Peat's bioenergetic insights*
"""
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        logger.info(f"✅ Processed {input_path.name} -> {noise_reduction:.1f}% noise reduction")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to process {input_path.name}: {e}")
        return False

def main():
    """Main processing function."""
    load_dotenv()
    
    # Configuration
    INPUT_DIR = Path("../../data/processed/cleaned_corpus_tier1")
    OUTPUT_DIR = Path("../../data/processed/ray_peat_signal")
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        return
    
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extractor = EnhancedSignalExtractor(api_key)
    
    # Process files
    input_files = list(INPUT_DIR.glob("*.txt"))
    logger.info(f"Found {len(input_files)} files to process")
    
    processed = 0
    failed = 0
    
    for input_file in input_files:
        output_file = OUTPUT_DIR / f"{input_file.stem}_signal.txt"
        
        # Skip if already processed and not testing
        if output_file.exists():
            logger.info(f"⏭️  Skipping {input_file.name} (already processed)")
            continue
        
        if process_file(input_file, output_file, extractor):
            processed += 1
        else:
            failed += 1
        
        # Progress update
        if (processed + failed) % 10 == 0:
            logger.info(f"Progress: {processed} processed, {failed} failed")
    
    logger.info(f"🎉 Complete! {processed} files processed, {failed} failed")

if __name__ == "__main__":
    main() 