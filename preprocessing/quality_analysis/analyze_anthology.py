#!/usr/bin/env python3
"""
Script to analyze Ray Peat Anthology Excel file and identify content
that hasn't been processed yet compared to existing raw data.
"""

import pandas as pd
import os
import glob
from pathlib import Path
import re
from urllib.parse import urlparse
import json

def read_excel_file(filepath):
    """Read the Ray Peat Anthology Excel file and return all sheets"""
    try:
        # Try to read all sheets
        excel_file = pd.ExcelFile(filepath)
        print(f"Found sheets: {excel_file.sheet_names}")
        
        all_data = {}
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            all_data[sheet_name] = df
            print(f"\nSheet '{sheet_name}' has {len(df)} rows and {len(df.columns)} columns")
            print(f"Columns: {list(df.columns)}")
            
        return all_data
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def get_existing_files():
    """Get list of all existing processed files"""
    raw_data_path = "data/raw/raw_data"
    existing_files = []
    
    # Walk through all directories and collect file names
    for root, dirs, files in os.walk(raw_data_path):
        for file in files:
            if file.endswith(('.txt', '.html', '.md')):
                # Get relative path from raw_data
                rel_path = os.path.relpath(os.path.join(root, file), raw_data_path)
                existing_files.append(file)
                
    return existing_files

def extract_urls_and_references(data_dict):
    """Extract URLs and references from the Excel data"""
    all_references = []
    
    for sheet_name, df in data_dict.items():
        print(f"\nAnalyzing sheet: {sheet_name}")
        
        # Look for columns that might contain URLs or references
        for col in df.columns:
            print(f"  Column: {col}")
            
            # Check for URL-like content
            for idx, cell_value in enumerate(df[col]):
                if pd.notna(cell_value):
                    cell_str = str(cell_value)
                    
                    # Look for URLs
                    if 'http' in cell_str.lower() or 'www.' in cell_str.lower():
                        all_references.append({
                            'sheet': sheet_name,
                            'column': col,
                            'row': idx + 1,
                            'type': 'url',
                            'content': cell_str,
                            'title': None
                        })
                    
                    # Look for file references or titles
                    elif len(cell_str) > 10 and any(keyword in cell_str.lower() for keyword in 
                                                  ['ray peat', 'interview', 'transcript', 'newsletter', 'article']):
                        all_references.append({
                            'sheet': sheet_name,
                            'column': col,
                            'row': idx + 1,
                            'type': 'reference',
                            'content': cell_str,
                            'title': cell_str
                        })
    
    return all_references

def find_duplicates(references, existing_files):
    """Find which references might already exist in processed data"""
    duplicates = []
    new_content = []
    
    for ref in references:
        content = ref['content']
        is_duplicate = False
        
        # Simple matching - look for similar titles or content
        for existing_file in existing_files:
            # Remove file extensions and normalize names
            existing_clean = re.sub(r'\.(txt|html|md)$', '', existing_file.lower())
            existing_clean = re.sub(r'[^a-z0-9\s]', ' ', existing_clean)
            existing_clean = ' '.join(existing_clean.split())
            
            content_clean = re.sub(r'[^a-z0-9\s]', ' ', content.lower())
            content_clean = ' '.join(content_clean.split())
            
            # Check for substantial overlap
            if len(content_clean) > 10:
                words_content = set(content_clean.split())
                words_existing = set(existing_clean.split())
                
                if len(words_content) > 0:
                    overlap = len(words_content.intersection(words_existing)) / len(words_content)
                    if overlap > 0.6:  # 60% overlap threshold
                        duplicates.append({
                            'reference': ref,
                            'existing_file': existing_file,
                            'overlap': overlap
                        })
                        is_duplicate = True
                        break
        
        if not is_duplicate:
            new_content.append(ref)
    
    return duplicates, new_content

def main():
    # Read the Excel file
    excel_data = read_excel_file("data/raw/Ray Peat Anthology.xlsx")
    if not excel_data:
        print("Failed to read Excel file")
        return
    
    # Get existing files
    existing_files = get_existing_files()
    print(f"\nFound {len(existing_files)} existing processed files")
    
    # Extract references from Excel
    references = extract_urls_and_references(excel_data)
    print(f"\nFound {len(references)} references in Excel file")
    
    # Find duplicates and new content
    duplicates, new_content = find_duplicates(references, existing_files)
    
    print(f"\nSummary:")
    print(f"Total references in Excel: {len(references)}")
    print(f"Likely duplicates: {len(duplicates)}")
    print(f"New content to process: {len(new_content)}")
    
    # Save detailed results
    results = {
        'summary': {
            'total_references': len(references),
            'duplicates': len(duplicates),
            'new_content': len(new_content),
            'existing_files_count': len(existing_files)
        },
        'duplicates': duplicates,
        'new_content': new_content,
        'all_references': references
    }
    
    with open('anthology_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed analysis saved to 'anthology_analysis.json'")
    
    # Print some examples
    print(f"\nFirst 5 new content items:")
    for i, item in enumerate(new_content[:5]):
        print(f"{i+1}. [{item['type']}] {item['content'][:100]}...")
    
    if duplicates:
        print(f"\nFirst 5 likely duplicates:")
        for i, dup in enumerate(duplicates[:5]):
            print(f"{i+1}. Excel: {dup['reference']['content'][:50]}...")
            print(f"    -> Existing: {dup['existing_file']} (overlap: {dup['overlap']:.2f})")

if __name__ == "__main__":
    main() 