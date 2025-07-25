"""
Rules-based cleaning functions for high-quality documents.
Handles HTML extraction, whitespace normalization, and artifact removal.
"""

import re
import html
from bs4 import BeautifulSoup
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_html(file_path):
    """
    Extract clean text content from HTML files.
    
    Args:
        file_path (str): Path to the HTML file
        
    Returns:
        str: Cleaned plain text content
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'head', 'meta', 'link']):
            element.decompose()
        
        # Extract text from semantic elements
        semantic_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'blockquote', 'article', 'main', 'section']
        
        extracted_text = []
        
        # Try to find semantic content first
        for tag in semantic_tags:
            elements = soup.find_all(tag)
            for element in elements:
                text = element.get_text().strip()
                if text and len(text) > 10:  # Filter out very short elements
                    extracted_text.append(text)
        
        # If no semantic content found, extract all text
        if not extracted_text:
            extracted_text = [soup.get_text()]
        
        # Join and decode HTML entities
        clean_text = '\n\n'.join(extracted_text)
        clean_text = html.unescape(clean_text)
        
        logger.info(f"Extracted {len(clean_text)} characters from HTML: {Path(file_path).name}")
        return clean_text
        
    except Exception as e:
        logger.error(f"Error cleaning HTML file {file_path}: {e}")
        return ""

def normalize_whitespace(text):
    """
    Normalize whitespace in text content.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with normalized whitespace
    """
    if not text:
        return ""
    
    # Collapse multiple spaces and tabs into single spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    
    # Collapse more than two consecutive newlines into exactly two
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive leading/trailing whitespace
    text = text.strip()
    
    logger.debug(f"Normalized whitespace: {len(text)} characters")
    return text

def remove_known_artifacts(text, file_path=""):
    """
    Remove common artifacts identified in the corpus analysis.
    
    Args:
        text (str): Input text
        file_path (str): Path to the file for context-specific cleaning
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    original_length = len(text)
    
    # Common header/footer patterns identified in analysis
    artifacts = [
        r'Townsend Letter for Doctors.*?\n',
        r'©.*?\d{4}.*?\n',
        r'Copyright.*?\d{4}.*?\n',
        r'All rights reserved.*?\n',
        r'raypeat\.com.*?\n',
        r'Ray Peat.*?Newsletter.*?\n',
        r'^\s*Page \d+.*?\n',
        r'^\s*\d+\s*$',  # Page numbers on their own line
        r'^\s*[-=]{3,}.*?\n',  # Horizontal rules
        r'Click here.*?\n',
        r'Read more.*?\n',
        r'Subscribe.*?\n',
        r'Email.*?@.*?\n',
    ]
    
    for pattern in artifacts:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # File-specific cleaning based on category
    file_path_str = str(file_path)
    if "Townsend" in file_path_str:
        text = re.sub(r'.*Townsend Letter.*?\n', '', text, flags=re.IGNORECASE)
    
    if "kmud" in file_path_str.lower():
        text = re.sub(r'KMUD.*?\n', '', text, flags=re.IGNORECASE)
    
    # Clean up after artifact removal
    text = normalize_whitespace(text)
    
    removed_chars = original_length - len(text)
    if removed_chars > 0:
        logger.info(f"Removed {removed_chars} characters of artifacts from {Path(file_path).name}")
    
    return text

def fix_common_ocr_errors(text):
    """
    Fix common OCR errors without AI.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with basic OCR errors fixed
    """
    if not text:
        return ""
    
    # Common OCR error patterns
    ocr_fixes = {
        r'\bscien- tific\b': 'scientific',
        r'\bosteo- porosis\b': 'osteoporosis',
        r'\bthyro- id\b': 'thyroid',
        r'\bmetab- olism\b': 'metabolism',
        r'\bhorm- one\b': 'hormone',
        r'\bprogester- one\b': 'progesterone',
        r'\bestro- gen\b': 'estrogen',
        r'\bprotein\s+s\b': 'proteins',
        r'\bvitamin\s+s\b': 'vitamins',
        r'\bcalci- um\b': 'calcium',
        r'\bmagnes- ium\b': 'magnesium',
    }
    
    fixes_applied = 0
    for pattern, replacement in ocr_fixes.items():
        new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if new_text != text:
            fixes_applied += 1
            text = new_text
    
    if fixes_applied > 0:
        logger.info(f"Applied {fixes_applied} OCR fixes")
    
    return text

def extract_metadata(text, file_path):
    """
    Extract basic metadata from text content.
    
    Args:
        text (str): Input text
        file_path (str): Original file path
        
    Returns:
        dict: Extracted metadata
    """
    metadata = {
        'original_file': file_path,
        'word_count': len(text.split()) if text else 0,
        'character_count': len(text) if text else 0,
        'extracted_title': '',
        'extracted_date': ''
    }
    
    if text:
        # Try to extract title (first non-empty line or heading)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            # Look for likely title patterns
            for line in lines[:5]:  # Check first 5 lines
                if 20 <= len(line) <= 200 and not line.endswith('.'):
                    metadata['extracted_title'] = line
                    break
            
            if not metadata['extracted_title'] and lines:
                metadata['extracted_title'] = lines[0][:100]  # Fallback to first line
        
        # Try to extract dates
        date_patterns = [
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b\d{1,2}-\d{1,2}-\d{4}\b'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                metadata['extracted_date'] = match.group()
                break
    
    return metadata

def clean_text_file(file_path):
    """
    Clean a plain text file using rules-based methods.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        str: Cleaned text content
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Apply text cleaning steps
        content = normalize_whitespace(content)
        content = remove_known_artifacts(content, file_path)
        content = fix_common_ocr_errors(content)
        
        logger.info(f"Cleaned text file: {Path(file_path).name} ({len(content)} characters)")
        return content
        
    except Exception as e:
        logger.error(f"Error cleaning text file {file_path}: {e}")
        return ""

if __name__ == "__main__":
    # Test the functions
    print("Rules-based cleaners module loaded successfully") 