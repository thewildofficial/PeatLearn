#!/usr/bin/env python3
"""
Calculate realistic time estimates for Ray Peat Anthology processing.
"""

import json
from urllib.parse import urlparse
from collections import defaultdict

def analyze_processing_plan():
    """Analyze what will actually be processed and estimate times"""
    
    with open('urls_for_processing.json', 'r') as f:
        data = json.load(f)
    
    all_urls = data['all_urls']
    
    # Categorize exactly like the comprehensive processor does
    categories = {
        'highest_value': [],  # raypeat.com, archive.org
        'high_value': [],     # forums, known sources  
        'medium_value': [],   # articles, PDFs, audio
        'video_content': [],  # videos (mostly will fail)
        'low_value': []       # everything else
    }
    
    HIGH_VALUE_DOMAINS = [
        'raypeat.com', 'web.archive.org', 'raypeatforum.com',
        'data.raypeatforum.com', 'www.functionalps.com', 
        'www.toxinless.com', 'wiki.chadnet.org'
    ]
    
    for url_data in all_urls:
        domain = urlparse(url_data['url']).netloc.lower()
        content_type = url_data.get('content_type', 'other')
        
        if content_type == 'video':
            categories['video_content'].append(url_data)
        elif 'raypeat.com' in domain or 'web.archive.org' in domain:
            categories['highest_value'].append(url_data)
        elif any(high_domain in domain for high_domain in HIGH_VALUE_DOMAINS):
            categories['high_value'].append(url_data)
        elif content_type in ['article', 'pdf', 'audio']:
            categories['medium_value'].append(url_data)
        else:
            categories['low_value'].append(url_data)
    
    return categories

def estimate_processing_times(categories):
    """Estimate processing times based on content types and success rates"""
    
    # Processing plan from the comprehensive script
    processing_plan = [
        ('highest_value', len(categories['highest_value']), None, 95),      # Process all, 95% success rate
        ('high_value', len(categories['high_value']), 100, 70),             # Process first 100, 70% success  
        ('medium_value', len(categories['medium_value']), 50, 60),          # Process first 50, 60% success
    ]
    
    total_urls = 0
    total_time_minutes = 0
    successful_downloads = 0
    
    print("📊 PROCESSING PLAN & TIME ESTIMATES:")
    print("=" * 60)
    
    for category, total_available, max_process, success_rate in processing_plan:
        urls_to_process = min(total_available, max_process) if max_process else total_available
        
        if urls_to_process == 0:
            continue
            
        # Time calculation: 3 seconds per URL (2s rate limit + 1s processing)
        time_minutes = (urls_to_process * 3) / 60
        expected_successful = int(urls_to_process * success_rate / 100)
        
        total_urls += urls_to_process
        total_time_minutes += time_minutes
        successful_downloads += expected_successful
        
        print(f"\n📂 {category.upper()}:")
        print(f"   Available: {total_available} URLs")
        print(f"   Processing: {urls_to_process} URLs")
        print(f"   Expected success rate: {success_rate}%")
        print(f"   Expected successful downloads: {expected_successful}")
        print(f"   Time estimate: {time_minutes:.1f} minutes ({time_minutes/60:.1f} hours)")
    
    print(f"\n🎯 TOTAL ESTIMATES:")
    print(f"   URLs to process: {total_urls}")
    print(f"   Expected successful downloads: {successful_downloads}")
    print(f"   Total time: {total_time_minutes:.1f} minutes ({total_time_minutes/60:.1f} hours)")
    
    return {
        'total_urls': total_urls,
        'total_time_minutes': total_time_minutes,
        'expected_successful': successful_downloads
    }

def main():
    print("RAY PEAT ANTHOLOGY - REALISTIC TIME ESTIMATE")
    print("=" * 60)
    
    categories = analyze_processing_plan()
    
    print("📋 CONTENT BREAKDOWN:")
    for category, urls in categories.items():
        print(f"   {category}: {len(urls)} URLs")
    
    estimates = estimate_processing_times(categories)
    
    # Current time estimate
    hours = estimates['total_time_minutes'] / 60
    
    print(f"\n⏰ COMPLETION ESTIMATE:")
    if hours < 1:
        print(f"   Estimated completion: {estimates['total_time_minutes']:.0f} minutes")
    else:
        print(f"   Estimated completion: {hours:.1f} hours")
    
    print(f"\n📈 EXPECTED RESULTS:")
    print(f"   New files added to corpus: ~{estimates['expected_successful']}")
    print(f"   Corpus expansion: Significant increase in Ray Peat content")
    
    print(f"\n💡 NOTES:")
    print(f"   • YouTube videos have low success rate (no transcripts)")
    print(f"   • raypeat.com content has highest success rate")
    print(f"   • Rate limiting ensures respectful scraping")
    print(f"   • Processing running in background - check with monitor_processing.py")

if __name__ == "__main__":
    main() 