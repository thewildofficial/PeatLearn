"""
Rules-based cleaning functions for Tier 1 (high quality) documents.
These functions handle HTML parsing, whitespace normalization, and removal of known artifacts.
"""

import re
import html
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_html(file_path):
    """
    Extract clean text from HTML files using BeautifulSoup.
    
    Args:
        file_path (str): Path to the HTML file
        
    Returns:
        str: Cleaned plain text content
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove unwanted elements
        unwanted_tags = ['head', 'script', 'style', 'nav', 'footer', 'iframe', 'noscript']
        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Extract text from semantic tags only
        semantic_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'blockquote', 'article', 'section', 'main', 'div']
        extracted_text = []
        
        # Try to find main content area first
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        if main_content:
            soup = main_content
        
        # Extract text from semantic elements
        for tag_name in semantic_tags:
            elements = soup.find_all(tag_name)
            for element in elements:
                text = element.get_text(separator=' ', strip=True)
                if text and len(text.strip()) > 10:  # Filter out very short texts
                    extracted_text.append(text)
        
        # If no semantic tags found, extract all text
        if not extracted_text:
            extracted_text = [soup.get_text(separator=' ', strip=True)]
        
        # Join all extracted text
        clean_text = '\n\n'.join(extracted_text)
        
        # Decode HTML entities
        clean_text = html.unescape(clean_text)
        
        logger.info(f"Extracted {len(clean_text)} characters from HTML file: {file_path}")
        return clean_text
        
    except Exception as e:
        logger.error(f"Error processing HTML file {file_path}: {str(e)}")
        return ""

def normalize_whitespace(text):
    """
    Normalize whitespace in text using regular expressions.
    
    Args:
        text (str): Input text with irregular whitespace
        
    Returns:
        str: Text with normalized whitespace
    """
    if not text:
        return ""
    
    # Collapse multiple spaces and tabs into single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove leading/trailing whitespace from each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    
    # Collapse more than two consecutive newlines into maximum of two
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading and trailing whitespace from entire text
    text = text.strip()
    
    logger.info(f"Normalized whitespace for text of {len(text)} characters")
    return text

def remove_known_artifacts(text, notes=""):
    """
    Remove known artifacts and headers/footers based on patterns identified in analysis.
    
    Args:
        text (str): Input text
        notes (str): Notes from the analysis CSV to guide artifact removal
        
    Returns:
        str: Text with artifacts removed
    """
    if not text:
        return ""
    
    # Common artifacts to remove based on corpus analysis
    common_artifacts = [
        r'Townsend Letter for Doctors.*?\d{4}',  # Publication headers
        r'Ray Peat\'?s Newsletter.*?(?:\d{4})?',  # Newsletter headers
        r'Page \d+',  # Page numbers
        r'^\d+$',  # Standalone page numbers
        r'From the original article.*?Ray Peat',  # Attribution lines
        r'https?://[^\s]+',  # URLs (optional - might want to preserve some)
        r'<!DOCTYPE.*?>',  # Any remaining HTML
        r'<[^>]+>',  # Any remaining HTML tags
        r'&[a-zA-Z]+;',  # HTML entities that weren't decoded
        r'^\s*[-=_]{3,}\s*$',  # Separator lines
        r'Copyright.*?\d{4}',  # Copyright notices
        r'All rights reserved',  # Rights notices
        r'^\s*\d+\s*$',  # Lines with only numbers
    ]
    
    # Apply artifact removal patterns
    for pattern in common_artifacts:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove specific artifacts mentioned in notes
    if notes:
        # Extract specific patterns from notes if they contain artifact descriptions
        if "Townsend Letter" in notes:
            text = re.sub(r'Townsend Letter.*?(?:\d{4}|\n)', '', text, flags=re.IGNORECASE)
        
        if "HTML entities" in notes:
            # Additional HTML entity cleanup
            text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
        
        if "navigation" in notes.lower():
            # Remove common navigation text
            nav_patterns = [
                r'Home\s*\|\s*About\s*\|\s*Contact',
                r'Next\s*\|\s*Previous',
                r'Table of Contents',
                r'Back to top',
            ]
            for pattern in nav_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Final cleanup
    text = normalize_whitespace(text)
    
    logger.info(f"Removed artifacts from text, final length: {len(text)} characters")
    return text

def clean_txt_file(file_path):
    """
    Clean a plain text file by reading and applying basic normalization.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        str: Cleaned text content
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        
        logger.info(f"Read {len(content)} characters from text file: {file_path}")
        return content
        
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {str(e)}")
        return ""

def extract_metadata(text, original_file_path):
    """
    Extract basic metadata from cleaned text.
    
    Args:
        text (str): Cleaned text content
        original_file_path (str): Path to original file
        
    Returns:
        dict: Extracted metadata
    """
    metadata = {
        'word_count': len(text.split()),
        'character_count': len(text),
        'estimated_reading_time': len(text.split()) / 200,  # Assume 200 WPM
    }
    
    # Try to extract title (first substantial line)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        # Look for title-like patterns
        for line in lines[:5]:  # Check first 5 lines
            if len(line) < 200 and len(line) > 10:  # Reasonable title length
                metadata['title'] = line
                break
    
    # Try to extract date patterns
    date_patterns = [
        r'\b(19|20)\d{2}\b',  # Year
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(19|20)\d{2}\b',
        r'\b\d{1,2}[-/]\d{1,2}[-/](19|20)?\d{2}\b',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text[:1000])  # Search in first 1000 chars
        if match:
            metadata['date'] = match.group()
            break
    
    return metadata 