#!/usr/bin/env python3
"""
Extract URLs and new content from Ray Peat Anthology for processing.
This script focuses on identifying actionable items that need to be scraped/processed.
"""

import pandas as pd
import json
import re
from urllib.parse import urlparse
from pathlib import Path

# Constants for filtering and prioritization
HIGH_PRIORITY_DOMAINS = [
    'raypeat.com',
    'youtube.com', 
    'youtu.be',
    'vimeo.com',
    'soundcloud.com'
]

CONTENT_TYPE_PRIORITY = {
    'video': 1,
    'audio': 2, 
    'article': 3,
    'pdf': 4,
    'other': 5
}

def extract_urls_from_excel():
    """Extract URLs directly from the Excel file with better parsing"""
    excel_path = "../../data/raw/Ray Peat Anthology.xlsx"
    
    try:
        # Focus on the main sheets with URLs
        items_df = pd.read_excel(excel_path, sheet_name='Items')
        deduplicated_df = pd.read_excel(excel_path, sheet_name='Deduplicated')
        
        urls = []
        
        # Extract from Items sheet
        for idx, row in items_df.iterrows():
            if pd.notna(row['Url']):
                url_data = {
                    'url': str(row['Url']).strip(),
                    'title': str(row['Title']) if pd.notna(row['Title']) else '',
                    'source': str(row['Source']) if pd.notna(row['Source']) else '',
                    'filetype': str(row['Filetype']) if pd.notna(row['Filetype']) else '',
                    'sheet': 'Items',
                    'row': idx + 2
                }
                
                if is_valid_url(url_data['url']):
                    urls.append(url_data)
        
        # Extract from Deduplicated sheet
        for idx, row in deduplicated_df.iterrows():
            for link_col in ['Link 1', 'Link 2', 'Link 3']:
                if pd.notna(row[link_col]):
                    url_data = {
                        'url': str(row[link_col]).strip(),
                        'title': str(row['Title']) if pd.notna(row['Title']) else '',
                        'source': 'Deduplicated',
                        'filetype': str(row['File Format']) if pd.notna(row['File Format']) else '',
                        'sheet': 'Deduplicated',
                        'row': idx + 2
                    }
                    
                    if is_valid_url(url_data['url']):
                        urls.append(url_data)
        
        return urls
    
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

def is_valid_url(url):
    """Check if URL is valid and processable"""
    if not url or len(url) < 10:
        return False
    
    # Skip obviously invalid URLs
    invalid_patterns = [
        'javascript:',
        'mailto:',
        'tel:',
        'file://',
        'localhost',
        '127.0.0.1'
    ]
    
    url_lower = url.lower()
    return not any(pattern in url_lower for pattern in invalid_patterns)

def categorize_content_type(url, filetype, title):
    """Determine content type for prioritization"""
    url_lower = url.lower()
    filetype_lower = filetype.lower() if filetype else ''
    title_lower = title.lower() if title else ''
    
    # Video content
    if any(domain in url_lower for domain in ['youtube.com', 'youtu.be', 'vimeo.com']):
        return 'video'
    
    # Audio content
    if any(domain in url_lower for domain in ['soundcloud.com']) or \
       any(term in filetype_lower for term in ['mp3', 'audio', 'wav']):
        return 'audio'
    
    # PDF content
    if 'pdf' in filetype_lower or url_lower.endswith('.pdf'):
        return 'pdf'
    
    # Articles/text content
    if any(term in title_lower for term in ['article', 'newsletter', 'blog', 'interview']):
        return 'article'
    
    return 'other'

def calculate_priority_score(url_data):
    """Calculate priority score for processing order"""
    score = 0
    url = url_data['url']
    domain = urlparse(url).netloc.lower()
    
    # High priority domains
    if any(priority_domain in domain for priority_domain in HIGH_PRIORITY_DOMAINS):
        score += 10
    
    # Content type priority
    content_type = categorize_content_type(url, url_data['filetype'], url_data['title'])
    score += (6 - CONTENT_TYPE_PRIORITY.get(content_type, 5))
    
    # Ray Peat official content gets highest priority
    if 'raypeat.com' in domain:
        score += 20
    
    # Penalize very long URLs (might be temporary/session-based)
    if len(url) > 200:
        score -= 5
    
    return score

def filter_existing_content(urls):
    """Filter out URLs that might already be processed"""
    # Simple check - could be enhanced with more sophisticated matching
    existing_files_path = "../../data/raw/raw_data"
    
    try:
        existing_files = []
        # Fix: Use os.walk instead of Path.rglob to avoid unpacking issues
        import os
        for root, dirs, files in os.walk(existing_files_path):
            for file in files:
                if file.endswith(('.txt', '.html', '.md')):
                    # Get just the filename without extension
                    file_stem = os.path.splitext(file)[0].lower()
                    existing_files.append(file_stem)
        
        filtered_urls = []
        for url_data in urls:
            title_clean = re.sub(r'[^a-z0-9\s]', '', url_data['title'].lower())
            
            # Check if title has significant overlap with existing files
            is_duplicate = False
            for existing in existing_files:
                if len(title_clean) > 10:
                    words_title = set(title_clean.split())
                    words_existing = set(existing.replace('-', ' ').replace('_', ' ').split())
                    
                    if len(words_title) > 0:
                        overlap = len(words_title.intersection(words_existing)) / len(words_title)
                        if overlap > 0.7:  # 70% overlap threshold
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                filtered_urls.append(url_data)
        
        return filtered_urls
    
    except Exception as e:
        print(f"Error filtering existing content: {e}")
        return urls

def main():
    print("Extracting URLs from Ray Peat Anthology...")
    
    # Extract URLs from Excel
    urls = extract_urls_from_excel()
    print(f"Found {len(urls)} URLs in anthology")
    
    # Filter out likely duplicates
    new_urls = filter_existing_content(urls)
    print(f"After filtering duplicates: {len(new_urls)} URLs remain")
    
    # Add content categorization and priority scores
    for url_data in new_urls:
        url_data['content_type'] = categorize_content_type(
            url_data['url'], 
            url_data['filetype'], 
            url_data['title']
        )
        url_data['priority_score'] = calculate_priority_score(url_data)
    
    # Sort by priority score (highest first)
    new_urls.sort(key=lambda x: x['priority_score'], reverse=True)
    
    # Group by content type for reporting
    by_type = {}
    for url_data in new_urls:
        content_type = url_data['content_type']
        if content_type not in by_type:
            by_type[content_type] = []
        by_type[content_type].append(url_data)
    
    # Generate summary report
    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total URLs to process: {len(new_urls)}")
    print(f"\nBreakdown by content type:")
    for content_type, urls_list in by_type.items():
        print(f"  {content_type}: {len(urls_list)} URLs")
    
    # Save prioritized URLs for processing
    output_data = {
        'summary': {
            'total_urls': len(new_urls),
            'by_content_type': {k: len(v) for k, v in by_type.items()}
        },
        'high_priority_urls': [url for url in new_urls if url['priority_score'] >= 15],
        'medium_priority_urls': [url for url in new_urls if 10 <= url['priority_score'] < 15],
        'low_priority_urls': [url for url in new_urls if url['priority_score'] < 10],
        'all_urls': new_urls
    }
    
    with open('urls_for_processing.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n=== PRIORITY BREAKDOWN ===")
    print(f"High priority (score ≥15): {len(output_data['high_priority_urls'])}")
    print(f"Medium priority (10-14): {len(output_data['medium_priority_urls'])}")
    print(f"Low priority (<10): {len(output_data['low_priority_urls'])}")
    
    print(f"\nTop 10 highest priority URLs:")
    for i, url_data in enumerate(new_urls[:10]):
        domain = urlparse(url_data['url']).netloc
        print(f"{i+1:2d}. [{url_data['priority_score']:2d}] {domain} - {url_data['title'][:50]}...")
    
    print(f"\nResults saved to 'urls_for_processing.json'")

if __name__ == "__main__":
    main() 