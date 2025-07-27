#!/usr/bin/env python3
"""
Web scraper for Ray Peat content from anthology URLs.
Handles different content types and saves to appropriate raw data folders.
"""

import requests
import json
import time
import re
from pathlib import Path
from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup
import yt_dlp
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContentScraper:
    def __init__(self, base_output_dir: str = "../../data/raw/raw_data"):
        self.base_output_dir = Path(base_output_dir)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Rate limiting
        self.delay_between_requests = 2.0
        self.last_request_time = 0
        
        # Statistics
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
    
    def _rate_limit(self):
        """Implement rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.delay_between_requests:
            time.sleep(self.delay_between_requests - time_since_last)
        self.last_request_time = time.time()
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system storage"""
        # Remove/replace problematic characters
        filename = re.sub(r'[<>:"/\\|?*]', '-', filename)
        filename = re.sub(r'\s+', ' ', filename)
        filename = filename.strip()
        
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        
        return filename
    
    def _determine_output_folder(self, url_data: Dict) -> Path:
        """Determine appropriate output folder based on content type"""
        content_type = url_data.get('content_type', 'other')
        source = url_data.get('source', 'Unknown')
        
        if content_type == 'video':
            return self.base_output_dir / "01_Audio_Transcripts" / "Video_Content"
        elif content_type == 'audio':
            return self.base_output_dir / "01_Audio_Transcripts" / "Audio_Content"
        elif content_type == 'pdf':
            return self.base_output_dir / "02_Publications" / "PDF_Documents"
        elif content_type == 'article':
            return self.base_output_dir / "02_Publications" / "Articles"
        elif content_type == 'email':
            return self.base_output_dir / "06_Email_Communications"
        else:
            return self.base_output_dir / "09_Miscellaneous"
    
    def scrape_webpage(self, url: str) -> Optional[Dict]:
        """Scrape text content from a webpage"""
        try:
            self._rate_limit()
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Extract main content
            # Try common content containers first
            content_selectors = [
                'article', 'main', '.content', '#content', 
                '.post', '.entry', '.article-content'
            ]
            
            content_element = None
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    break
            
            if not content_element:
                content_element = soup.find('body')
            
            if content_element:
                text = content_element.get_text()
                # Clean up text
                text = re.sub(r'\n\s*\n', '\n\n', text)
                text = re.sub(r'[ \t]+', ' ', text)
                text = text.strip()
                
                return {
                    'title': title_text,
                    'content': text,
                    'url': url,
                    'content_type': 'text/html'
                }
        
        except Exception as e:
            logger.error(f"Error scraping webpage {url}: {e}")
            return None
    
    def download_youtube_transcript(self, url: str) -> Optional[Dict]:
        """Download YouTube video transcript"""
        try:
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en'],
                'skip_download': True,
                'quiet': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # Try to get transcript
                if 'subtitles' in info and 'en' in info['subtitles']:
                    subtitle_url = info['subtitles']['en'][0]['url']
                    response = self.session.get(subtitle_url)
                    transcript = response.text
                    
                    return {
                        'title': info.get('title', ''),
                        'content': transcript,
                        'url': url,
                        'content_type': 'youtube_transcript',
                        'description': info.get('description', ''),
                        'duration': info.get('duration', 0)
                    }
        
        except Exception as e:
            logger.error(f"Error downloading YouTube transcript {url}: {e}")
            return None
    
    def process_url(self, url_data: Dict) -> bool:
        """Process a single URL and save content"""
        url = url_data['url']
        title = url_data.get('title', 'Untitled')
        content_type = url_data.get('content_type', 'other')
        
        logger.info(f"Processing: {title[:50]}... ({content_type})")
        
        # Determine scraping method based on URL
        domain = urlparse(url).netloc.lower()
        content = None
        
        if 'youtube.com' in domain or 'youtu.be' in domain:
            content = self.download_youtube_transcript(url)
        else:
            content = self.scrape_webpage(url)
        
        if not content:
            logger.warning(f"Failed to extract content from {url}")
            return False
        
        # Determine output location
        output_folder = self._determine_output_folder(url_data)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        filename = self._sanitize_filename(title)
        if not filename:
            filename = f"content_{int(time.time())}"
        
        # Add appropriate extension
        if content['content_type'] == 'youtube_transcript':
            filename += '.mp3-transcript.txt'
        else:
            filename += '.html'
        
        output_path = output_folder / filename
        
        # Save content
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Title: {content['title']}\n")
                f.write(f"URL: {url}\n")
                f.write(f"Content Type: {content['content_type']}\n")
                f.write("=" * 80 + "\n\n")
                f.write(content['content'])
            
            logger.info(f"Saved: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving content to {output_path}: {e}")
            return False
    
    def process_url_list(self, urls_file: str, max_urls: Optional[int] = None, priority_filter: str = 'high'):
        """Process a list of URLs from the JSON file"""
        try:
            with open(urls_file, 'r') as f:
                data = json.load(f)
            
            # Select URLs based on priority
            if priority_filter == 'high':
                urls_to_process = data['high_priority_urls']
            elif priority_filter == 'medium':
                urls_to_process = data['medium_priority_urls']
            elif priority_filter == 'low':
                urls_to_process = data['low_priority_urls']
            else:
                urls_to_process = data['all_urls']
            
            if max_urls:
                urls_to_process = urls_to_process[:max_urls]
            
            logger.info(f"Starting to process {len(urls_to_process)} URLs ({priority_filter} priority)")
            
            for i, url_data in enumerate(urls_to_process, 1):
                logger.info(f"Progress: {i}/{len(urls_to_process)}")
                
                self.stats['processed'] += 1
                
                try:
                    if self.process_url(url_data):
                        self.stats['successful'] += 1
                    else:
                        self.stats['failed'] += 1
                except Exception as e:
                    logger.error(f"Unexpected error processing URL {url_data['url']}: {e}")
                    self.stats['failed'] += 1
                
                # Save progress periodically
                if i % 10 == 0:
                    logger.info(f"Progress update: {self.stats}")
            
            logger.info(f"Processing complete. Final stats: {self.stats}")
        
        except Exception as e:
            logger.error(f"Error processing URL list: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape Ray Peat content from anthology URLs')
    parser.add_argument('--urls-file', default='urls_for_processing.json', 
                       help='JSON file containing URLs to process')
    parser.add_argument('--max-urls', type=int, help='Maximum number of URLs to process')
    parser.add_argument('--priority', choices=['high', 'medium', 'low', 'all'], 
                       default='high', help='Priority level of URLs to process')
    parser.add_argument('--output-dir', default='../../data/raw/raw_data',
                       help='Output directory for scraped content')
    
    args = parser.parse_args()
    
    scraper = ContentScraper(base_output_dir=args.output_dir)
    scraper.process_url_list(args.urls_file, args.max_urls, args.priority)

if __name__ == "__main__":
    main() 