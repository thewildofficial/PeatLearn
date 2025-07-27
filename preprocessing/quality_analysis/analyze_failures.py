#!/usr/bin/env python3
"""
Analyze common failure reasons for Ray Peat Anthology URLs.
Helps understand why some URLs fail and what can be improved.
"""

import json
import requests
from urllib.parse import urlparse
from collections import defaultdict, Counter
import time

def test_url_accessibility(url, timeout=10):
    """Test if a URL is accessible and why it might fail"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        
        return {
            'accessible': True,
            'status_code': response.status_code,
            'content_type': response.headers.get('Content-Type', 'unknown'),
            'final_url': response.url,
            'redirected': response.url != url
        }
    
    except requests.exceptions.Timeout:
        return {'accessible': False, 'error': 'timeout'}
    except requests.exceptions.ConnectionError:
        return {'accessible': False, 'error': 'connection_error'}
    except requests.exceptions.HTTPError as e:
        return {'accessible': False, 'error': f'http_error_{e.response.status_code}'}
    except Exception as e:
        return {'accessible': False, 'error': f'other_{type(e).__name__}'}

def analyze_failure_patterns():
    """Analyze patterns in failed URLs"""
    
    with open('urls_for_processing.json', 'r') as f:
        data = json.load(f)
    
    # Get a sample of high-priority URLs for testing
    high_priority = data['high_priority_urls'][:20]  # Test first 20
    
    results = {
        'accessible': [],
        'failed': [],
        'by_domain': defaultdict(list),
        'by_content_type': defaultdict(list),
        'error_types': Counter()
    }
    
    print("🔍 TESTING URL ACCESSIBILITY...")
    print("Testing first 20 high-priority URLs for common failure patterns\n")
    
    for i, url_data in enumerate(high_priority, 1):
        url = url_data['url']
        domain = urlparse(url).netloc
        content_type = url_data.get('content_type', 'unknown')
        
        print(f"[{i:2d}/20] Testing: {domain}")
        
        result = test_url_accessibility(url)
        result['url_data'] = url_data
        
        results['by_domain'][domain].append(result)
        results['by_content_type'][content_type].append(result)
        
        if result['accessible']:
            results['accessible'].append(result)
            print(f"         ✅ OK ({result['status_code']})")
        else:
            results['failed'].append(result)
            results['error_types'][result['error']] += 1
            print(f"         ❌ FAILED ({result['error']})")
        
        time.sleep(0.5)  # Be nice to servers
    
    return results

def identify_common_issues(results):
    """Identify the most common failure reasons"""
    
    print(f"\n📊 ACCESSIBILITY ANALYSIS RESULTS")
    print("=" * 50)
    
    total_tested = len(results['accessible']) + len(results['failed'])
    success_rate = len(results['accessible']) / total_tested * 100
    
    print(f"Success rate: {success_rate:.1f}% ({len(results['accessible'])}/{total_tested})")
    
    if results['failed']:
        print(f"\n❌ COMMON FAILURE REASONS:")
        for error_type, count in results['error_types'].most_common():
            print(f"   {error_type}: {count} URLs")
    
    # Analyze by domain
    print(f"\n🌐 RESULTS BY DOMAIN:")
    domain_success = {}
    for domain, domain_results in results['by_domain'].items():
        successful = sum(1 for r in domain_results if r['accessible'])
        total = len(domain_results)
        success_rate = successful / total * 100
        domain_success[domain] = success_rate
        print(f"   {domain}: {success_rate:.0f}% ({successful}/{total})")
    
    # Analyze by content type  
    print(f"\n📁 RESULTS BY CONTENT TYPE:")
    for content_type, content_results in results['by_content_type'].items():
        successful = sum(1 for r in content_results if r['accessible'])
        total = len(content_results)
        success_rate = successful / total * 100
        print(f"   {content_type}: {success_rate:.0f}% ({successful}/{total})")
    
    return domain_success

def suggest_improvements(results, domain_success):
    """Suggest improvements based on failure analysis"""
    
    print(f"\n💡 SUGGESTED IMPROVEMENTS:")
    
    # Check for common error patterns
    if 'timeout' in results['error_types']:
        print(f"   • Increase timeout for slow-loading sites")
    
    if 'connection_error' in results['error_types']:
        print(f"   • Some sites may be temporarily down or blocking requests")
    
    if any('http_error_4' in error for error in results['error_types']):
        print(f"   • Some URLs return 404/403 - may need better URL validation")
    
    # Domain-specific suggestions
    problematic_domains = [domain for domain, success in domain_success.items() if success < 50]
    if problematic_domains:
        print(f"   • Consider special handling for: {', '.join(problematic_domains)}")
    
    # Video content issues
    video_results = results['by_content_type'].get('video', [])
    if video_results:
        video_success = sum(1 for r in video_results if r['accessible']) / len(video_results) * 100
        if video_success < 50:
            print(f"   • YouTube videos have low success rate for transcript extraction")
            print(f"   • Consider alternative transcript sources or manual processing")
    
    print(f"\n🎯 PROCESSING RECOMMENDATIONS:")
    print(f"   • Focus on domains with >70% success rate first")
    print(f"   • Process videos separately with specialized tools")
    print(f"   • Implement retry logic for timeout errors")
    print(f"   • Add URL validation before processing")

def main():
    print("RAY PEAT ANTHOLOGY - FAILURE ANALYSIS")
    print("=" * 50)
    
    try:
        results = analyze_failure_patterns()
        domain_success = identify_common_issues(results)
        suggest_improvements(results, domain_success)
        
        print(f"\n✅ Analysis complete!")
        print(f"This analysis helps explain why some URLs fail during scraping.")
        
    except FileNotFoundError:
        print("❌ Error: urls_for_processing.json not found")
        print("Run extract_urls_for_processing.py first")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    main() 