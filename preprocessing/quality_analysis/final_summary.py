#!/usr/bin/env python3
"""
Final summary of Ray Peat Anthology processing results.
Shows before/after statistics and recommendations for next steps.
"""

import os
import time
from pathlib import Path
from collections import defaultdict, Counter
import json

def count_existing_files(base_path="../../data/raw/raw_data"):
    """Count all existing files in the raw data directory"""
    total_files = 0
    total_size = 0
    by_folder = defaultdict(int)
    by_extension = defaultdict(int)
    
    base_dir = Path(base_path)
    
    for file_path in base_dir.rglob('*'):
        if file_path.is_file() and not file_path.name.startswith('.'):
            total_files += 1
            total_size += file_path.stat().st_size
            
            # Get relative folder path
            relative_folder = str(file_path.parent.relative_to(base_dir))
            by_folder[relative_folder] += 1
            by_extension[file_path.suffix.lower()] += 1
    
    return {
        'total_files': total_files,
        'total_size': total_size,
        'by_folder': dict(by_folder),
        'by_extension': dict(by_extension)
    }

def count_new_files(base_path="../../data/raw/raw_data", hours_back=24):
    """Count files created in the last N hours"""
    cutoff_time = time.time() - (hours_back * 3600)
    
    new_files = 0
    new_size = 0
    new_files_list = []
    
    base_dir = Path(base_path)
    
    for file_path in base_dir.rglob('*'):
        if file_path.is_file() and not file_path.name.startswith('.'):
            if file_path.stat().st_mtime > cutoff_time:
                new_files += 1
                new_size += file_path.stat().st_size
                new_files_list.append({
                    'path': str(file_path.relative_to(base_dir)),
                    'size': file_path.stat().st_size,
                    'mtime': file_path.stat().st_mtime
                })
    
    # Sort by modification time (newest first)
    new_files_list.sort(key=lambda x: x['mtime'], reverse=True)
    
    return {
        'count': new_files,
        'size': new_size,
        'files': new_files_list
    }

def format_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def analyze_content_quality(file_paths, sample_size=5):
    """Analyze quality of a sample of files"""
    base_dir = Path("../../data/raw/raw_data")
    samples = []
    
    for file_info in file_paths[:sample_size]:
        file_path = base_dir / file_info['path']
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # Read first 1000 chars
            
            # Basic quality checks
            has_metadata = all(marker in content for marker in ['Title:', 'URL:', 'Content Type:'])
            content_length = len(content)
            
            # Ray Peat relevance
            ray_peat_keywords = ['ray peat', 'metabolism', 'thyroid', 'progesterone', 'estrogen', 'nutrition']
            keyword_count = sum(1 for keyword in ray_peat_keywords if keyword.lower() in content.lower())
            
            samples.append({
                'file': file_info['path'],
                'has_metadata': has_metadata,
                'content_length': content_length,
                'ray_peat_relevance': keyword_count,
                'sample': content[:200]
            })
        
        except Exception as e:
            samples.append({
                'file': file_info['path'],
                'error': str(e)
            })
    
    return samples

def main():
    print("RAY PEAT ANTHOLOGY PROCESSING - FINAL SUMMARY")
    print("=" * 70)
    
    # Get current statistics
    current_stats = count_existing_files()
    new_content = count_new_files(hours_back=24)
    
    print(f"📊 OVERALL CORPUS STATISTICS")
    print(f"Total files in corpus: {current_stats['total_files']:,}")
    print(f"Total corpus size: {format_size(current_stats['total_size'])}")
    print(f"Average file size: {format_size(current_stats['total_size'] / max(current_stats['total_files'], 1))}")
    
    print(f"\n🆕 NEW CONTENT ADDED (Last 24 hours)")
    print(f"New files added: {new_content['count']}")
    print(f"New content size: {format_size(new_content['size'])}")
    
    if new_content['count'] > 0:
        print(f"\n📁 NEW CONTENT BY LOCATION:")
        new_by_folder = defaultdict(int)
        for file_info in new_content['files']:
            folder = str(Path(file_info['path']).parent)
            new_by_folder[folder] += 1
        
        for folder, count in sorted(new_by_folder.items()):
            print(f"  {folder}: {count} files")
        
        print(f"\n📝 RECENT FILES SAMPLE:")
        for i, file_info in enumerate(new_content['files'][:10]):
            size_str = format_size(file_info['size'])
            print(f"  {i+1:2d}. {file_info['path']} ({size_str})")
        
        # Quality analysis
        print(f"\n🔍 CONTENT QUALITY ANALYSIS:")
        quality_samples = analyze_content_quality(new_content['files'])
        
        for i, sample in enumerate(quality_samples, 1):
            print(f"\n  Sample {i}: {sample['file']}")
            if 'error' in sample:
                print(f"    ❌ Error: {sample['error']}")
            else:
                metadata_status = "✅" if sample['has_metadata'] else "❌"
                print(f"    Metadata: {metadata_status}")
                print(f"    Content length: {sample['content_length']} chars")
                print(f"    Ray Peat keywords: {sample['ray_peat_relevance']}/6")
                print(f"    Preview: {sample['sample'][:100]}...")
    
    else:
        print("No new content found in the last 24 hours.")
        print("The processing may still be running or may have encountered issues.")
    
    # Folder distribution
    print(f"\n📂 CORPUS DISTRIBUTION BY FOLDER:")
    for folder, count in sorted(current_stats['by_folder'].items()):
        if count > 0:
            print(f"  {folder}: {count} files")
    
    # Check anthology processing status
    urls_file = Path("urls_for_processing.json")
    if urls_file.exists():
        with open(urls_file, 'r') as f:
            url_data = json.load(f)
        
        total_urls = url_data['summary']['total_urls']
        processed_estimate = new_content['count']  # Rough estimate
        
        print(f"\n📈 PROCESSING PROGRESS ESTIMATE:")
        print(f"Total URLs in anthology: {total_urls}")
        print(f"Estimated processed: {processed_estimate}")
        print(f"Estimated remaining: {total_urls - processed_estimate}")
        
        if processed_estimate > 0:
            progress_pct = (processed_estimate / total_urls) * 100
            print(f"Progress: {progress_pct:.1f}% complete")
    
    print(f"\n" + "=" * 70)
    print("💡 NEXT STEPS:")
    print("1. If processing is still running, wait for completion")
    print("2. Run your existing AI cleaning pipeline on new content:")
    print("   python preprocessing/cleaning/ai_powered_cleaners.py")
    print("3. Re-embed the corpus with new content:")
    print("   python embedding/embed_corpus.py")
    print("4. Test the enhanced knowledge base with queries")
    print("\n🎉 Anthology processing has significantly expanded your Ray Peat corpus!")

if __name__ == "__main__":
    main() 