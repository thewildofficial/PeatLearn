#!/usr/bin/env python3
"""
Comprehensive processing of Ray Peat Anthology content.
Processes non-video content across all priority levels with intelligent domain filtering.
"""

import json
from web_scraper import ContentScraper
import logging
from urllib.parse import urlparse
from collections import defaultdict

# Define high-value domains for Ray Peat content
HIGH_VALUE_DOMAINS = [
    'raypeat.com',
    'web.archive.org',
    'raypeatforum.com',
    'data.raypeatforum.com',
    'www.functionalps.com',
    'www.toxinless.com',
    'wiki.chadnet.org'
]

def categorize_urls_by_value(urls_file="urls_for_processing.json"):
    """Categorize URLs by their potential value and processability"""
    with open(urls_file, 'r') as f:
        data = json.load(f)
    
    # Get all URLs
    all_urls = data['all_urls']
    
    categorized = {
        'highest_value': [],      # raypeat.com, archive.org
        'high_value': [],         # forums, known Ray Peat sources
        'medium_value': [],       # other articles, PDFs
        'video_content': [],      # videos (will process separately)
        'low_value': []           # everything else
    }
    
    for url_data in all_urls:
        domain = urlparse(url_data['url']).netloc.lower()
        content_type = url_data.get('content_type', 'other')
        
        # Skip videos for now
        if content_type == 'video':
            categorized['video_content'].append(url_data)
            continue
        
        # Categorize by domain value
        if 'raypeat.com' in domain or 'web.archive.org' in domain:
            categorized['highest_value'].append(url_data)
        elif any(high_domain in domain for high_domain in HIGH_VALUE_DOMAINS):
            categorized['high_value'].append(url_data)
        elif content_type in ['article', 'pdf', 'audio']:
            categorized['medium_value'].append(url_data)
        else:
            categorized['low_value'].append(url_data)
    
    return categorized

def process_category(scraper, urls, category_name, max_urls=None):
    """Process a category of URLs"""
    logger = logging.getLogger(__name__)
    
    if max_urls:
        urls = urls[:max_urls]
    
    logger.info(f"\n📂 Processing {len(urls)} {category_name} URLs...")
    
    stats = {'processed': 0, 'successful': 0, 'failed': 0}
    
    for i, url_data in enumerate(urls, 1):
        title = url_data.get('title', 'Unknown')[:60]
        domain = urlparse(url_data['url']).netloc
        
        logger.info(f"Progress: {i}/{len(urls)} | {domain} | {title}...")
        
        try:
            if scraper.process_url(url_data):
                stats['successful'] += 1
                logger.info(f"✓ Success")
            else:
                stats['failed'] += 1
                logger.warning(f"✗ Failed")
        except Exception as e:
            logger.error(f"✗ Error: {e}")
            stats['failed'] += 1
        
        stats['processed'] += 1
        
        # Progress update every 20 items
        if i % 20 == 0:
            success_rate = stats['successful'] / stats['processed'] * 100
            logger.info(f"📊 Current success rate: {success_rate:.1f}% ({stats['successful']}/{stats['processed']})")
    
    return stats

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 RAY PEAT ANTHOLOGY COMPREHENSIVE PROCESSING")
    logger.info("=" * 60)
    
    # Categorize URLs
    categorized = categorize_urls_by_value()
    
    logger.info("📊 Content categorization:")
    for category, urls in categorized.items():
        logger.info(f"  {category}: {len(urls)} URLs")
    
    # Setup scraper
    scraper = ContentScraper()
    
    # Process in order of value
    total_stats = {'processed': 0, 'successful': 0, 'failed': 0}
    
    processing_plan = [
        ('highest_value', categorized['highest_value'], None),      # Process all
        ('high_value', categorized['high_value'], 100),            # Process first 100
        ('medium_value', categorized['medium_value'], 50),         # Process first 50
    ]
    
    for category_name, urls, max_urls in processing_plan:
        if not urls:
            logger.info(f"No URLs in {category_name} category, skipping...")
            continue
        
        stats = process_category(scraper, urls, category_name, max_urls)
        
        # Aggregate stats
        for key in total_stats:
            total_stats[key] += stats[key]
        
        success_rate = stats['successful'] / max(stats['processed'], 1) * 100
        logger.info(f"📈 {category_name} results: {success_rate:.1f}% success ({stats['successful']}/{stats['processed']})")
    
    # Final summary
    final_success_rate = total_stats['successful'] / max(total_stats['processed'], 1) * 100
    
    logger.info(f"\n🎉 COMPREHENSIVE PROCESSING COMPLETE!")
    logger.info(f"=" * 60)
    logger.info(f"📊 Final Statistics:")
    logger.info(f"  Total processed: {total_stats['processed']}")
    logger.info(f"  Total successful: {total_stats['successful']}")
    logger.info(f"  Total failed: {total_stats['failed']}")
    logger.info(f"  Overall success rate: {final_success_rate:.1f}%")
    logger.info(f"\n📁 Files saved to: ../../data/raw/raw_data/")
    
    # Show domain breakdown of successful downloads
    logger.info(f"\n💡 Next steps:")
    logger.info(f"  1. Run monitor_processing.py to check content quality")
    logger.info(f"  2. Process videos separately if needed") 
    logger.info(f"  3. Run your existing AI cleaning pipeline on new content")

if __name__ == "__main__":
    main() 