#!/usr/bin/env python3
"""
Process only text-based content for high success rates.
Uses the filtered processable URLs to avoid failures.
"""

import json
from web_scraper import ContentScraper
import logging

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 PROCESSING TEXT CONTENT ONLY")
    logger.info("=" * 60)
    
    # Load filtered processable URLs
    try:
        with open('processable_urls.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error("❌ processable_urls.json not found. Run filter_processable_urls.py first")
        return
    
    processable_urls = data['processable_urls']
    
    # Categorize by value (from the filtered data)
    categories = {
        'highest_value': [],
        'high_value': [],
        'medium_value': []
    }
    
    HIGH_VALUE_DOMAINS = ['raypeat.com', 'web.archive.org']
    MEDIUM_HIGH_DOMAINS = ['raypeatforum.com', 'data.raypeatforum.com', 'www.toxinless.com', 'wiki.chadnet.org']
    
    for url_data in processable_urls:
        domain = url_data['url'].lower()
        
        if any(hv_domain in domain for hv_domain in HIGH_VALUE_DOMAINS):
            categories['highest_value'].append(url_data)
        elif any(mh_domain in domain for mh_domain in MEDIUM_HIGH_DOMAINS):
            categories['high_value'].append(url_data)
        else:
            categories['medium_value'].append(url_data)
    
    logger.info(f"📊 PROCESSABLE CONTENT BREAKDOWN:")
    for category, urls in categories.items():
        logger.info(f"   {category}: {len(urls)} URLs")
    
    # Setup scraper
    scraper = ContentScraper()
    
    # Process in order of value
    total_stats = {'processed': 0, 'successful': 0, 'failed': 0}
    
    processing_plan = [
        ('highest_value', categories['highest_value'], None),      # Process all
        ('high_value', categories['high_value'], 50),             # Process first 50
        ('medium_value', categories['medium_value'], 25),         # Process first 25
    ]
    
    for category_name, urls, max_urls in processing_plan:
        if not urls:
            continue
            
        if max_urls:
            urls = urls[:max_urls]
        
        logger.info(f"\n📂 Processing {len(urls)} {category_name} URLs...")
        
        stats = {'processed': 0, 'successful': 0, 'failed': 0}
        
        for i, url_data in enumerate(urls, 1):
            title = url_data.get('title', 'Unknown')[:50]
            domain = url_data['url'].split('/')[2]
            
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
            
            # Progress update every 10 items
            if i % 10 == 0:
                success_rate = stats['successful'] / stats['processed'] * 100
                logger.info(f"📊 Current success rate: {success_rate:.1f}% ({stats['successful']}/{stats['processed']})")
        
        # Aggregate stats
        for key in total_stats:
            total_stats[key] += stats[key]
        
        success_rate = stats['successful'] / max(stats['processed'], 1) * 100
        logger.info(f"📈 {category_name} results: {success_rate:.1f}% success ({stats['successful']}/{stats['processed']})")
    
    # Final summary
    final_success_rate = total_stats['successful'] / max(total_stats['processed'], 1) * 100
    
    logger.info(f"\n🎉 TEXT CONTENT PROCESSING COMPLETE!")
    logger.info(f"=" * 60)
    logger.info(f"📊 Final Statistics:")
    logger.info(f"  Total processed: {total_stats['processed']}")
    logger.info(f"  Total successful: {total_stats['successful']}")
    logger.info(f"  Total failed: {total_stats['failed']}")
    logger.info(f"  Overall success rate: {final_success_rate:.1f}%")
    logger.info(f"\n📁 Files saved to: ../../data/raw/raw_data/")
    logger.info(f"✅ High-quality text content successfully added to your corpus!")

if __name__ == "__main__":
    main() 