#!/usr/bin/env python3
"""
Scrape email-related content from Ray Peat Anthology.
Focuses on email exchanges, correspondence, and advice.
"""

import json
from web_scraper import ContentScraper
import logging
from urllib.parse import urlparse

def find_email_content():
    """Find all email-related URLs from the anthology"""
    
    with open('urls_for_processing.json', 'r') as f:
        data = json.load(f)
    
    email_urls = []
    
    # Search criteria for email content
    email_keywords = [
        'email', 'correspondence', 'advice', 'depository', 
        'exchanges', 'letters', 'communication', 'wiki emails'
    ]
    
    for url_data in data['all_urls']:
        title = url_data.get('title', '').lower()
        url = url_data.get('url', '').lower()
        
        # Check title for email-related keywords
        if any(keyword in title for keyword in email_keywords):
            # Skip YouTube videos (likely won't have email content)
            if 'youtube.com' not in url and 'youtu.be' not in url:
                email_urls.append(url_data)
    
    return email_urls

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("📧 RAY PEAT EMAIL CONTENT SCRAPER")
    logger.info("=" * 50)
    
    # Find email-related URLs
    email_urls = find_email_content()
    
    if not email_urls:
        logger.info("❌ No email-related URLs found in anthology")
        return
    
    logger.info(f"📊 Found {len(email_urls)} email-related URLs:")
    for i, url_data in enumerate(email_urls, 1):
        title = url_data.get('title', 'No title')
        domain = urlparse(url_data['url']).netloc
        logger.info(f"  {i}. {domain} - {title}")
    
    # Setup scraper
    scraper = ContentScraper(base_output_dir="../../data/raw/raw_data")
    
    # Process each email URL
    stats = {'processed': 0, 'successful': 0, 'failed': 0}
    
    logger.info(f"\n🚀 Starting email content scraping...")
    
    for i, url_data in enumerate(email_urls, 1):
        title = url_data.get('title', 'Unknown')
        domain = urlparse(url_data['url']).netloc
        
        logger.info(f"\nProgress: {i}/{len(email_urls)}")
        logger.info(f"Processing: {title}")
        logger.info(f"URL: {url_data['url']}")
        
        try:
            # Override content type to ensure it goes to Email Communications folder
            url_data_copy = url_data.copy()
            url_data_copy['content_type'] = 'email'
            
            if scraper.process_url(url_data_copy):
                stats['successful'] += 1
                logger.info(f"✅ SUCCESS - Email content saved")
            else:
                stats['failed'] += 1
                logger.warning(f"❌ FAILED - Could not extract content")
        
        except Exception as e:
            stats['failed'] += 1
            logger.error(f"❌ ERROR - {e}")
        
        stats['processed'] += 1
    
    # Final summary
    success_rate = stats['successful'] / max(stats['processed'], 1) * 100
    
    logger.info(f"\n📊 EMAIL SCRAPING RESULTS:")
    logger.info(f"=" * 50)
    logger.info(f"Total processed: {stats['processed']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Success rate: {success_rate:.1f}%")
    
    if stats['successful'] > 0:
        logger.info(f"\n📁 Email content saved to:")
        logger.info(f"   ../../data/raw/raw_data/06_Email_Communications/")
        logger.info(f"\n✅ Successfully expanded your Ray Peat email collection!")
    else:
        logger.info(f"\n❌ No new email content was successfully extracted")

if __name__ == "__main__":
    main() 