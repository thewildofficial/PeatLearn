#!/usr/bin/env python3
"""
Monitor AI processing progress for new Ray Peat content.
"""

import os
import time
from pathlib import Path
from collections import defaultdict

def count_processed_files():
    """Count processed files in the AI cleaned directory"""
    base_path = Path("../../data/processed/ai_cleaned")
    
    processed_count = 0
    by_folder = defaultdict(int)
    
    for file_path in base_path.rglob("*_processed.txt"):
        processed_count += 1
        folder = str(file_path.parent.relative_to(base_path))
        by_folder[folder] += 1
    
    return processed_count, dict(by_folder)

def count_raw_files():
    """Count raw files that need processing"""
    base_path = Path("../../data/raw/raw_data")
    
    raw_count = 0
    new_files = []
    
    for file_path in base_path.rglob("*"):
        if file_path.is_file() and file_path.suffix in ['.txt', '.html', '.md']:
            raw_count += 1
            
            # Check if it's a recent file (last 24 hours)
            if time.time() - file_path.stat().st_mtime < 86400:
                new_files.append(str(file_path.relative_to(base_path)))
    
    return raw_count, new_files

def check_new_processed_folders():
    """Check for newly created processed folders"""
    base_path = Path("../../data/processed/ai_cleaned")
    
    new_folders = []
    for folder_path in base_path.iterdir():
        if folder_path.is_dir():
            folder_name = folder_path.name
            if folder_name in ['09_Miscellaneous', '02_Publications'] and folder_path.stat().st_mtime > time.time() - 3600:  # Last hour
                new_folders.append(folder_name)
    
    return new_folders

def main():
    print("🤖 AI PROCESSING MONITOR")
    print("=" * 50)
    
    # Count current state
    processed_count, by_folder = count_processed_files()
    raw_count, new_files = count_raw_files()
    new_folders = check_new_processed_folders()
    
    print(f"📊 CURRENT STATUS:")
    print(f"   Processed files: {processed_count}")
    print(f"   Raw files: {raw_count}")
    print(f"   Processing progress: {processed_count/max(raw_count,1)*100:.1f}%")
    
    if new_files:
        print(f"\n🆕 NEW FILES ADDED (last 24h): {len(new_files)}")
        for file in new_files[:10]:  # Show first 10
            print(f"   • {file}")
        if len(new_files) > 10:
            print(f"   ... and {len(new_files) - 10} more")
    
    if new_folders:
        print(f"\n📁 NEW PROCESSED FOLDERS:")
        for folder in new_folders:
            print(f"   ✅ {folder}")
    
    print(f"\n📂 PROCESSED BY FOLDER:")
    for folder, count in sorted(by_folder.items()):
        if count > 0:
            print(f"   {folder}: {count} files")
    
    # Check for new anthology content specifically
    anthology_files = [
        "../../data/processed/ai_cleaned/09_Miscellaneous",
        "../../data/processed/ai_cleaned/02_Publications/Articles"
    ]
    
    print(f"\n🔍 ANTHOLOGY CONTENT STATUS:")
    for folder_path in anthology_files:
        folder = Path(folder_path)
        if folder.exists():
            count = len(list(folder.glob("*_processed.txt")))
            print(f"   ✅ {folder.name}: {count} processed files")
        else:
            print(f"   ⏳ {folder.name}: Not yet processed")
    
    print(f"\n💡 To check detailed progress:")
    print(f"   cd ../../preprocessing/cleaning")
    print(f"   cat full_corpus_checkpoint.json")

if __name__ == "__main__":
    main() 