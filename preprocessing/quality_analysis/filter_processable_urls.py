#!/usr/bin/env python3
"""
Filter URLs to focus on processable text content.
Removes direct audio files, videos, and other non-text content.
"""

import json
from urllib.parse import urlparse
from collections import defaultdict

def is_processable_url(url):
    """Check if a URL points to processable text content"""
    url_lower = url.lower()
    
    # Skip direct audio/video files
    audio_extensions = ['.mp3', '.wav', '.m4a', '.aac', '.ogg']
    video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv']
    other_extensions = ['.pdf', '.doc', '.docx', '.zip', '.exe']
    
    # Check if URL ends with non-processable extensions
    for ext in audio_extensions + video_extensions + other_extensions:
        if url_lower.endswith(ext):
            return False, f'direct_file_{ext[1:]}'
    
    # Skip known problematic patterns
    if '/wp-content/uploads/' in url_lower and any(ext in url_lower for ext in audio_extensions):
        return False, 'direct_audio_file'
    
    # YouTube videos (we know these mostly fail for transcript extraction)
    if 'youtube.com/watch' in url_lower or 'youtu.be/' in url_lower:
        return False, 'youtube_video'
    
    return True, 'processable'

def filter_processable_content():
    """Filter the URLs to focus on processable text content"""
    
    with open('urls_for_processing.json', 'r') as f:
        data = json.load(f)
    
    all_urls = data['all_urls']
    
    processable = []
    filtered_out = []
    filter_reasons = defaultdict(int)
    
    for url_data in all_urls:
        url = url_data['url']
        is_proc, reason = is_processable_url(url)
        
        if is_proc:
            processable.append(url_data)
        else:
            filtered_out.append({**url_data, 'filter_reason': reason})
            filter_reasons[reason] += 1
    
    # Categorize processable URLs by value
    categories = {
        'highest_value': [],  # raypeat.com, archive.org
        'high_value': [],     # forums, known sources
        'medium_value': [],   # other articles
        'low_value': []       # everything else
    }
    
    HIGH_VALUE_DOMAINS = [
        'raypeat.com', 'web.archive.org', 'raypeatforum.com',
        'data.raypeatforum.com', 'www.functionalps.com', 
        'www.toxinless.com', 'wiki.chadnet.org'
    ]
    
    for url_data in processable:
        domain = urlparse(url_data['url']).netloc.lower()
        content_type = url_data.get('content_type', 'other')
        
        if 'raypeat.com' in domain or 'web.archive.org' in domain:
            categories['highest_value'].append(url_data)
        elif any(high_domain in domain for high_domain in HIGH_VALUE_DOMAINS):
            categories['high_value'].append(url_data)
        elif content_type in ['article', 'other'] and len(url_data.get('title', '')) > 10:
            categories['medium_value'].append(url_data)
        else:
            categories['low_value'].append(url_data)
    
    return processable, filtered_out, filter_reasons, categories

def main():
    print("RAY PEAT ANTHOLOGY - PROCESSABLE CONTENT FILTER")
    print("=" * 60)
    
    processable, filtered_out, filter_reasons, categories = filter_processable_content()
    
    print(f"📊 FILTERING RESULTS:")
    print(f"Original URLs: {len(processable) + len(filtered_out)}")
    print(f"Processable URLs: {len(processable)}")
    print(f"Filtered out: {len(filtered_out)}")
    
    print(f"\n❌ FILTERED OUT BY REASON:")
    for reason, count in sorted(filter_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"   {reason}: {count} URLs")
    
    print(f"\n✅ PROCESSABLE CONTENT BY VALUE:")
    total_processable = 0
    for category, urls in categories.items():
        if urls:
            print(f"   {category}: {len(urls)} URLs")
            total_processable += len(urls)
    
    print(f"\n🎯 RECOMMENDED PROCESSING:")
    processing_plan = [
        ('highest_value', len(categories['highest_value']), None),
        ('high_value', len(categories['high_value']), 50),
        ('medium_value', len(categories['medium_value']), 25),
    ]
    
    total_to_process = 0
    expected_success = 0
    
    for category, available, max_process in processing_plan:
        if available > 0:
            to_process = min(available, max_process) if max_process else available
            success_rate = 95 if category == 'highest_value' else (80 if category == 'high_value' else 70)
            expected = int(to_process * success_rate / 100)
            
            total_to_process += to_process
            expected_success += expected
            
            print(f"   {category}: {to_process} URLs → ~{expected} successful")
    
    time_estimate = (total_to_process * 3) / 60  # 3 seconds per URL
    
    print(f"\n⏰ REALISTIC ESTIMATES:")
    print(f"   URLs to process: {total_to_process}")
    print(f"   Expected successful downloads: ~{expected_success}")
    print(f"   Estimated time: {time_estimate:.1f} minutes")
    
    # Save filtered results
    filtered_data = {
        'summary': {
            'total_processable': len(processable),
            'by_category': {k: len(v) for k, v in categories.items()}
        },
        'processable_urls': processable,
        'filtered_out': filtered_out[:10],  # Save first 10 as examples
        'filter_reasons': dict(filter_reasons)
    }
    
    with open('processable_urls.json', 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"\n📁 Results saved to 'processable_urls.json'")
    print(f"💡 Use this filtered list for much higher success rates!")

if __name__ == "__main__":
    main() 