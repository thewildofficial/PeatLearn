#!/usr/bin/env python3
"""
Process non-video content from Ray Peat Anthology.
Focuses on articles, PDFs, and audio content which have higher success rates.
"""

import json
from web_scraper import ContentScraper
import logging

def filter_non_video_urls(urls_file="urls_for_processing.json"):
    """Filter out video URLs to focus on more processable content"""
    with open(urls_file, 'r') as f:
        data = json.load(f)
    
    # Get all high and medium priority URLs
    all_priority_urls = data['high_priority_urls'] + data['medium_priority_urls']
    
    # Filter out video content
    non_video_urls = []
    for url_data in all_priority_urls:
        content_type = url_data.get('content_type', 'other')
        if content_type != 'video':
            non_video_urls.append(url_data)
    
    return non_video_urls

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Get non-video URLs
    non_video_urls = filter_non_video_urls()
    
    logger.info(f"Found {len(non_video_urls)} non-video URLs to process")
    
    # Categorize by content type
    by_type = {}
    for url_data in non_video_urls:
        content_type = url_data.get('content_type', 'other')
        if content_type not in by_type:
            by_type[content_type] = []
        by_type[content_type].append(url_data)
    
    logger.info("Content breakdown:")
    for content_type, urls in by_type.items():
        logger.info(f"  {content_type}: {len(urls)} URLs")
    
    # Process with scraper
    scraper = ContentScraper()
    
    total_processed = 0
    total_successful = 0
    
    # Process each content type
    for content_type, urls in by_type.items():
        logger.info(f"\nProcessing {len(urls)} {content_type} URLs...")
        
        for i, url_data in enumerate(urls, 1):
            logger.info(f"Progress: {i}/{len(urls)} ({content_type})")
            
            try:
                if scraper.process_url(url_data):
                    total_successful += 1
                    logger.info(f"✓ Successfully processed: {url_data.get('title', 'Unknown')[:50]}...")
                else:
                    logger.warning(f"✗ Failed to process: {url_data.get('title', 'Unknown')[:50]}...")
                
                total_processed += 1
                
                # Progress update every 10 items
                if i % 10 == 0:
                    success_rate = total_successful / total_processed * 100
                    logger.info(f"Current success rate: {success_rate:.1f}% ({total_successful}/{total_processed})")
            
            except Exception as e:
                logger.error(f"Error processing {url_data.get('url', 'Unknown URL')}: {e}")
                total_processed += 1
    
    # Final statistics
    final_success_rate = total_successful / max(total_processed, 1) * 100
    logger.info(f"\n🎉 PROCESSING COMPLETE!")
    logger.info(f"Total processed: {total_processed}")
    logger.info(f"Total successful: {total_successful}")
    logger.info(f"Success rate: {final_success_rate:.1f}%")
    logger.info(f"Files saved to: ../../data/raw/raw_data/")

if __name__ == "__main__":
    main() 