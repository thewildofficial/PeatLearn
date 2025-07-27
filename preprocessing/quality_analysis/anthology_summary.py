#!/usr/bin/env python3
"""
Summary script for Ray Peat Anthology analysis.
Provides an overview of findings and recommendations for next steps.
"""

import json
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter, defaultdict

def load_analysis_data():
    """Load the analysis results"""
    try:
        with open('urls_for_processing.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: urls_for_processing.json not found. Run extract_urls_for_processing.py first.")
        return None

def analyze_domains(urls):
    """Analyze domain distribution"""
    domains = []
    for url_data in urls:
        domain = urlparse(url_data['url']).netloc.lower()
        domains.append(domain)
    
    return Counter(domains)

def analyze_content_types(urls):
    """Analyze content type distribution"""
    types = [url_data.get('content_type', 'unknown') for url_data in urls]
    return Counter(types)

def get_highest_priority_samples(data, n=10):
    """Get samples of highest priority content"""
    high_priority = data.get('high_priority_urls', [])
    return high_priority[:n]

def generate_processing_recommendations(data):
    """Generate recommendations for processing strategy"""
    recommendations = []
    
    total_urls = data['summary']['total_urls']
    high_priority_count = len(data.get('high_priority_urls', []))
    
    recommendations.append(f"📊 ANALYSIS SUMMARY:")
    recommendations.append(f"   • Total new URLs found: {total_urls:,}")
    recommendations.append(f"   • High priority (Ray Peat official + key domains): {high_priority_count}")
    recommendations.append(f"   • Estimated processing time: {total_urls * 3 / 60:.1f} hours")
    
    recommendations.append(f"\n🎯 RECOMMENDED PROCESSING STRATEGY:")
    recommendations.append(f"   1. Start with HIGH PRIORITY ({high_priority_count} URLs)")
    recommendations.append(f"      - raypeat.com content (official articles)")
    recommendations.append(f"      - YouTube videos with transcripts")
    recommendations.append(f"      - Key interview sources")
    
    recommendations.append(f"   2. Process in batches of 50-100 URLs")
    recommendations.append(f"   3. Use rate limiting (2 seconds between requests)")
    recommendations.append(f"   4. Monitor for duplicate content during processing")
    
    # Analyze domains for specific recommendations
    all_urls = data.get('all_urls', [])
    domains = analyze_domains(all_urls)
    
    recommendations.append(f"\n🌐 TOP DOMAINS TO PROCESS:")
    for domain, count in domains.most_common(10):
        recommendations.append(f"   • {domain}: {count} URLs")
    
    content_types = analyze_content_types(all_urls)
    recommendations.append(f"\n📁 CONTENT TYPE BREAKDOWN:")
    for content_type, count in content_types.most_common():
        recommendations.append(f"   • {content_type}: {count} URLs")
    
    return recommendations

def generate_command_examples():
    """Generate example commands for processing"""
    commands = []
    
    commands.append("🚀 PROCESSING COMMANDS:")
    commands.append("")
    commands.append("# Test with first 5 high-priority URLs:")
    commands.append("python web_scraper.py --priority high --max-urls 5")
    commands.append("")
    commands.append("# Process all high-priority URLs:")
    commands.append("python web_scraper.py --priority high")
    commands.append("")
    commands.append("# Process high + medium priority:")
    commands.append("python web_scraper.py --priority medium")
    commands.append("")
    commands.append("# Process specific batch size:")
    commands.append("python web_scraper.py --priority high --max-urls 50")
    
    return commands

def show_sample_content(data):
    """Show samples of what will be processed"""
    samples = get_highest_priority_samples(data, 15)
    
    print("🔍 SAMPLE HIGH-PRIORITY CONTENT TO BE PROCESSED:")
    print("=" * 80)
    
    for i, url_data in enumerate(samples, 1):
        domain = urlparse(url_data['url']).netloc
        title = url_data.get('title', 'No title')[:60]
        content_type = url_data.get('content_type', 'unknown')
        priority = url_data.get('priority_score', 0)
        
        print(f"{i:2d}. [{priority:2d}] {content_type:8s} | {domain:20s} | {title}...")

def main():
    print("RAY PEAT ANTHOLOGY ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Load data
    data = load_analysis_data()
    if not data:
        return
    
    # Generate and display recommendations
    recommendations = generate_processing_recommendations(data)
    for rec in recommendations:
        print(rec)
    
    print("\n" + "=" * 60)
    
    # Show sample content
    show_sample_content(data)
    
    print("\n" + "=" * 60)
    
    # Show command examples
    commands = generate_command_examples()
    for cmd in commands:
        print(cmd)
    
    print("\n" + "=" * 60)
    print("📋 NEXT STEPS:")
    print("1. Review the sample content above")
    print("2. Start with a small test batch (5-10 URLs)")
    print("3. Check output quality in data/raw/raw_data/")
    print("4. Scale up processing based on results")
    print("5. Monitor for duplicate content and adjust filters")
    print("\n⚠️  IMPORTANT: Always test with small batches first!")

if __name__ == "__main__":
    main() 