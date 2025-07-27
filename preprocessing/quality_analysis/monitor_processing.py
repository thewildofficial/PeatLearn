#!/usr/bin/env python3
"""
Monitor the progress of Ray Peat Anthology content processing.
Shows statistics on downloaded files and processing quality.
"""

import os
import time
from pathlib import Path
from collections import defaultdict, Counter
import json

def scan_new_content(base_path="../../data/raw/raw_data"):
    """Scan for newly added content"""
    base_dir = Path(base_path)
    
    # Define the new folders created by the scraper
    new_folders = [
        "01_Audio_Transcripts/Video_Content",
        "01_Audio_Transcripts/Audio_Content", 
        "02_Publications/PDF_Documents",
        "02_Publications/Articles",
        "09_Miscellaneous"
    ]
    
    stats = {
        'total_files': 0,
        'by_folder': defaultdict(int),
        'by_extension': defaultdict(int),
        'total_size': 0,
        'files_by_size': [],
        'recent_files': []
    }
    
    current_time = time.time()
    
    for folder in new_folders:
        folder_path = base_dir / folder
        if folder_path.exists():
            for file_path in folder_path.rglob('*'):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    file_stat = file_path.stat()
                    file_size = file_stat.st_size
                    file_mtime = file_stat.st_mtime
                    
                    # Only count files modified in the last 24 hours as "new"
                    if current_time - file_mtime < 86400:  # 24 hours
                        stats['total_files'] += 1
                        stats['by_folder'][folder] += 1
                        stats['by_extension'][file_path.suffix.lower()] += 1
                        stats['total_size'] += file_size
                        
                        stats['files_by_size'].append((file_path, file_size))
                        stats['recent_files'].append((file_path, file_mtime))
    
    # Sort by size and recency
    stats['files_by_size'].sort(key=lambda x: x[1], reverse=True)
    stats['recent_files'].sort(key=lambda x: x[1], reverse=True)
    
    return stats

def format_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def format_time_ago(timestamp):
    """Format time difference in human readable format"""
    diff = time.time() - timestamp
    
    if diff < 60:
        return f"{int(diff)} seconds ago"
    elif diff < 3600:
        return f"{int(diff/60)} minutes ago"
    elif diff < 86400:
        return f"{int(diff/3600)} hours ago"
    else:
        return f"{int(diff/86400)} days ago"

def check_content_quality(file_path, max_sample_size=500):
    """Quick quality check of content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(max_sample_size)
            
        # Basic quality indicators
        has_title = 'Title:' in content
        has_url = 'URL:' in content
        has_content_type = 'Content Type:' in content
        content_length = len(content)
        
        # Look for Ray Peat related keywords
        ray_peat_keywords = ['ray peat', 'metabolism', 'thyroid', 'progesterone', 'estrogen']
        keyword_count = sum(1 for keyword in ray_peat_keywords if keyword.lower() in content.lower())
        
        return {
            'has_metadata': has_title and has_url and has_content_type,
            'content_length': content_length,
            'ray_peat_relevance': keyword_count,
            'sample_content': content[:200] + "..." if len(content) > 200 else content
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    print("RAY PEAT ANTHOLOGY PROCESSING MONITOR")
    print("=" * 60)
    
    # Scan for new content
    stats = scan_new_content()
    
    if stats['total_files'] == 0:
        print("No new files found in the last 24 hours.")
        print("The scraper may not be running or hasn't finished processing yet.")
        return
    
    print(f"📊 PROCESSING STATISTICS")
    print(f"Total new files: {stats['total_files']}")
    print(f"Total size: {format_size(stats['total_size'])}")
    print(f"Average file size: {format_size(stats['total_size'] / max(stats['total_files'], 1))}")
    
    print(f"\n📁 FILES BY FOLDER:")
    for folder, count in stats['by_folder'].items():
        print(f"  {folder}: {count} files")
    
    print(f"\n📄 FILES BY TYPE:")
    for ext, count in stats['by_extension'].items():
        ext_display = ext if ext else "(no extension)"
        print(f"  {ext_display}: {count} files")
    
    print(f"\n⏰ MOST RECENT FILES:")
    for i, (file_path, mtime) in enumerate(stats['recent_files'][:10]):
        relative_path = str(file_path).replace(str(Path("../../data/raw/raw_data").resolve()), "")
        print(f"  {i+1:2d}. {relative_path} ({format_time_ago(mtime)})")
    
    print(f"\n📏 LARGEST FILES:")
    for i, (file_path, size) in enumerate(stats['files_by_size'][:5]):
        relative_path = str(file_path).replace(str(Path("../../data/raw/raw_data").resolve()), "")
        print(f"  {i+1}. {relative_path} ({format_size(size)})")
    
    # Quality check on a few recent files
    print(f"\n🔍 CONTENT QUALITY SAMPLES:")
    for i, (file_path, _) in enumerate(stats['recent_files'][:3]):
        print(f"\n  File {i+1}: {file_path.name}")
        quality = check_content_quality(file_path)
        
        if 'error' in quality:
            print(f"    Error reading file: {quality['error']}")
        else:
            print(f"    Has metadata: {'✓' if quality['has_metadata'] else '✗'}")
            print(f"    Content length: {quality['content_length']} chars")
            print(f"    Ray Peat keywords: {quality['ray_peat_relevance']}/5")
            print(f"    Sample: {quality['sample_content'][:100]}...")
    
    print(f"\n" + "=" * 60)
    print("💡 To continue processing:")
    print("   python web_scraper.py --priority high --max-urls 50")
    print("\n💡 To process all remaining high priority:")
    print("   python web_scraper.py --priority high")

if __name__ == "__main__":
    main() 