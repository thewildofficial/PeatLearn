"""
AI-powered cleaning functions for low-quality documents.
Uses Gemini API for OCR correction, document segmentation, and speaker attribution.
"""

import json
import time
import logging
import re
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AICleaningError(Exception):
    """Custom exception for AI cleaning errors"""
    pass

def initialize_gemini():
    """Initialize Gemini API with error handling."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise AICleaningError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    return model

def make_api_call_with_retry(model, prompt, max_retries=3, delay=1):
    """
    Make API call with exponential backoff retry logic.
    
    Args:
        model: Gemini model instance
        prompt (str): Prompt to send to API
        max_retries (int): Maximum number of retries
        delay (int): Initial delay in seconds
        
    Returns:
        str: API response text
    """
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
            else:
                logger.warning(f"Empty response from API, attempt {attempt + 1}")
                
        except Exception as e:
            logger.error(f"API call failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                raise AICleaningError(f"API call failed after {max_retries} attempts: {e}")
    
    raise AICleaningError("API returned empty response after all retry attempts")

def correct_ocr_errors(text, model=None):
    """
    Correct OCR errors in text using AI.
    
    Args:
        text (str): Text with OCR errors
        model: Gemini model instance (optional)
        
    Returns:
        str: Corrected text
    """
    if not text or len(text.strip()) < 10:
        return text
    
    if model is None:
        model = initialize_gemini()
    
    # Limit text size to avoid token limits
    max_chars = 15000
    if len(text) > max_chars:
        logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars}")
        text = text[:max_chars] + "..."
    
    prompt = f"""You are an expert proofreader AI. The following text is from an OCR scan and contains many errors. 

Your task:
1. Correct all spelling and grammar errors
2. Fix merged words (e.g., "thethyroid" → "the thyroid")
3. Fix split words (e.g., "thyro id" → "thyroid")
4. Normalize whitespace and formatting
5. Preserve the original meaning and structure
6. Do NOT add new content or interpretations

Return only the corrected text without any explanations or comments.

Text to correct:
{text}"""

    try:
        corrected_text = make_api_call_with_retry(model, prompt)
        
        # Basic validation
        if len(corrected_text) < len(text) * 0.5:
            logger.warning("Corrected text suspiciously short, using original")
            return text
        
        logger.info(f"OCR correction: {len(text)} → {len(corrected_text)} characters")
        return corrected_text
        
    except Exception as e:
        logger.error(f"OCR correction failed: {e}")
        return text

def segment_document(text, model=None):
    """
    Segment a multi-document text into individual articles.
    
    Args:
        text (str): Text containing multiple documents
        model: Gemini model instance (optional)
        
    Returns:
        list: List of dictionaries with 'title' and 'content' keys
    """
    if not text or len(text.strip()) < 50:
        return [{"title": "Document", "content": text}]
    
    if model is None:
        model = initialize_gemini()
    
    # Limit text size
    max_chars = 20000
    if len(text) > max_chars:
        logger.warning(f"Text too long for segmentation ({len(text)} chars), truncating")
        text = text[:max_chars] + "..."
    
    prompt = f"""You are a document analysis AI. The following text contains multiple, unrelated articles or documents that need to be separated.

Your task:
1. Identify distinct articles/documents within the text
2. Create a descriptive title for each article (maximum 10 words)
3. Extract the complete content for each article
4. Return the result as a valid JSON array

Format:
[
  {{
    "title": "Brief descriptive title",
    "content": "Complete article text..."
  }},
  {{
    "title": "Another article title", 
    "content": "Complete article text..."
  }}
]

Important:
- If the text appears to be a single document, return it as one item
- Titles should be descriptive but concise
- Include all original content without adding interpretations
- Ensure valid JSON format

Text to segment:
{text}"""

    try:
        response = make_api_call_with_retry(model, prompt)
        
        # Clean JSON response
        clean_response = response.strip()
        if clean_response.startswith('```json'):
            clean_response = clean_response[7:]
        if clean_response.endswith('```'):
            clean_response = clean_response[:-3]
        clean_response = clean_response.strip()
        
        # Parse JSON
        segments = json.loads(clean_response)
        
        if not isinstance(segments, list) or not segments:
            logger.warning("Invalid segmentation result, returning original as single segment")
            return [{"title": "Document", "content": text}]
        
        # Validate segments
        valid_segments = []
        for i, segment in enumerate(segments):
            if isinstance(segment, dict) and 'title' in segment and 'content' in segment:
                if len(segment['content'].strip()) > 10:  # Minimum content length
                    valid_segments.append({
                        'title': str(segment['title'])[:100],  # Limit title length
                        'content': str(segment['content'])
                    })
        
        if not valid_segments:
            logger.warning("No valid segments found, returning original")
            return [{"title": "Document", "content": text}]
        
        logger.info(f"Segmented document into {len(valid_segments)} parts")
        return valid_segments
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse segmentation JSON: {e}")
        return [{"title": "Document", "content": text}]
    except Exception as e:
        logger.error(f"Document segmentation failed: {e}")
        return [{"title": "Document", "content": text}]

def attribute_speakers(text, model=None):
    """
    Attribute speakers in transcript text.
    
    Args:
        text (str): Transcript text without clear speaker attribution
        model: Gemini model instance (optional)
        
    Returns:
        str: Formatted transcript with speaker attribution
    """
    if not text or len(text.strip()) < 20:
        return text
    
    if model is None:
        model = initialize_gemini()
    
    # Limit text size
    max_chars = 18000
    if len(text) > max_chars:
        logger.warning(f"Transcript too long ({len(text)} chars), truncating")
        text = text[:max_chars] + "..."
    
    prompt = f"""You are an expert in transcript analysis. The following text is a conversation or interview transcript that needs better speaker attribution.

Your task:
1. Identify different speakers in the conversation
2. Format each paragraph with clear speaker labels (e.g., "Interviewer:", "Ray Peat:", "Host:", etc.)
3. Remove conversational fillers like "um", "uh", "you know"
4. Clean up grammar while preserving the conversational tone
5. Maintain the original meaning and content

Guidelines:
- Use consistent speaker labels throughout
- If you can't identify speakers clearly, use "Speaker 1:", "Speaker 2:", etc.
- Keep Ray Peat's responses labeled as "Ray Peat:" when identifiable
- Format each speaker turn on a new line
- Remove repetitive verbal tics but keep the natural flow

Return only the formatted transcript without explanations.

Transcript to format:
{text}"""

    try:
        formatted_text = make_api_call_with_retry(model, prompt)
        
        # Basic validation
        if len(formatted_text) < len(text) * 0.4:
            logger.warning("Formatted transcript suspiciously short, using original")
            return text
        
        logger.info(f"Speaker attribution: {len(text)} → {len(formatted_text)} characters")
        return formatted_text
        
    except Exception as e:
        logger.error(f"Speaker attribution failed: {e}")
        return text

def enhance_text_quality(text, model=None):
    """
    General text quality enhancement for problematic documents.
    
    Args:
        text (str): Text needing quality improvement
        model: Gemini model instance (optional)
        
    Returns:
        str: Enhanced text
    """
    if not text or len(text.strip()) < 20:
        return text
    
    if model is None:
        model = initialize_gemini()
    
    # Limit text size
    max_chars = 16000
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    
    prompt = f"""You are a text quality expert. The following text has various quality issues that need to be addressed.

Your task:
1. Fix formatting and spacing issues
2. Correct obvious errors without changing meaning
3. Improve readability while preserving original content
4. Remove redundant repetitions
5. Standardize punctuation and capitalization

Important:
- Do NOT add new information or interpretations
- Preserve all technical terms and names exactly
- Maintain the original structure and flow
- Focus on cleaning rather than rewriting

Return only the cleaned text.

Text to enhance:
{text}"""

    try:
        enhanced_text = make_api_call_with_retry(model, prompt)
        
        # Validation
        if len(enhanced_text) < len(text) * 0.5:
            logger.warning("Enhanced text too short, using original")
            return text
        
        logger.info(f"Text enhancement: {len(text)} → {len(enhanced_text)} characters")
        return enhanced_text
        
    except Exception as e:
        logger.error(f"Text enhancement failed: {e}")
        return text

def get_cleaning_strategy(semantic_score, atomicity_score, fidelity_score, speaker_score, file_path=""):
    """
    Determine the best AI cleaning strategy based on quality scores.
    
    Args:
        semantic_score (float): Semantic noise score
        atomicity_score (float): Document atomicity score  
        fidelity_score (float): Textual fidelity score
        speaker_score (float): Speaker attribution score (may be None)
        file_path (str): File path for context
        
    Returns:
        str: Recommended cleaning strategy
    """
    strategies = []
    
    # Prioritize by severity of issues
    if atomicity_score >= 7:
        strategies.append("segment")
    
    if fidelity_score >= 7:
        strategies.append("ocr_correct")
    
    if speaker_score and speaker_score >= 6:
        strategies.append("speaker_attribution")
    
    if semantic_score >= 8:
        strategies.append("enhance")
    
    # Default strategy for moderate issues
    if not strategies:
        if fidelity_score >= 5:
            strategies.append("ocr_correct")
        elif semantic_score >= 6:
            strategies.append("enhance")
        else:
            strategies.append("enhance")  # Fallback
    
    return strategies[0] if strategies else "enhance"

if __name__ == "__main__":
    # Test the functions
    try:
        model = initialize_gemini()
        print("AI-powered cleaners module loaded successfully")
        print("Gemini API connection established")
    except Exception as e:
        print(f"Error initializing AI cleaners: {e}") 