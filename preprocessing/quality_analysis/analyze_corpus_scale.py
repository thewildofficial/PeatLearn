#!/usr/bin/env python3
"""
Analyze processed corpus scale and token counts.
Note: Token counting uses a general-purpose tokenizer proxy.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict
import tiktoken

def count_tokens(text: str) -> int:
    """Count tokens using a fast tokenizer proxy."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base") # Assuming a default model for the proxy
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback: rough estimate (1 token ≈ 4 characters for English)
        return len(text) // 4

def parse_ray_peat_file(file_path):
    """Parse a single Ray Peat processed file and extract Q&A pairs"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    pairs = []
    
    # Split by RAY PEAT and CONTEXT markers
    # Pattern matches: **RAY PEAT:** or **CONTEXT:**
    sections = re.split(r'\*\*(RAY PEAT|CONTEXT):\*\*', content)
    
    current_pair = {}
    
    for i in range(1, len(sections), 2):
        if i + 1 < len(sections):
            marker = sections[i].strip()
            text = sections[i + 1].strip()
            
            if marker == "RAY PEAT" and text:
                current_pair['ray_peat'] = text
            elif marker == "CONTEXT" and text:
                current_pair['context'] = text
                
                # When we have both context and ray_peat, save the pair
                if 'ray_peat' in current_pair and current_pair['ray_peat']:
                    pairs.append({
                        'context': current_pair['context'],
                        'ray_peat': current_pair['ray_peat'],
                        'combined_text': f"QUESTION: {current_pair['context']}\n\nRAY PEAT: {current_pair['ray_peat']}"
                    })
                current_pair = {}
    
    return pairs

def analyze_directory(directory_path):
    """Analyze all processed files in a directory"""
    directory_stats = {
        'file_count': 0,
        'total_pairs': 0,
        'total_tokens': 0,
        'files': []
    }
    
    for file_path in Path(directory_path).rglob('*.txt'):
        if file_path.name == 'README.md':
            continue
            
        pairs = parse_ray_peat_file(file_path)
        
        file_tokens = 0
        for pair in pairs:
            file_tokens += count_tokens(pair['combined_text'])
        
        file_stats = {
            'name': str(file_path.relative_to(directory_path)),
            'size_kb': file_path.stat().st_size / 1024,
            'pair_count': len(pairs),
            'token_count': file_tokens,
            'avg_tokens_per_pair': file_tokens / len(pairs) if pairs else 0
        }
        
        directory_stats['files'].append(file_stats)
        directory_stats['file_count'] += 1
        directory_stats['total_pairs'] += len(pairs)
        directory_stats['total_tokens'] += file_tokens
    
    return directory_stats

def calculate_embedding_costs(total_tokens):
    """Calculate embedding costs for different providers"""
    costs = {}
    
    # Gemini embedding costs (estimated based on current pricing)
    costs['gemini'] = {
        'cost_per_1k_tokens': 0.0001,  # Estimated - verify current pricing
        'total_cost': (total_tokens / 1000) * 0.0001
    }
    
    # OpenAI text-embedding-3-small
    costs['openai_small'] = {
        'cost_per_1k_tokens': 0.00002,
        'total_cost': (total_tokens / 1000) * 0.00002
    }
    
    # OpenAI text-embedding-3-large  
    costs['openai_large'] = {
        'cost_per_1k_tokens': 0.00013,
        'total_cost': (total_tokens / 1000) * 0.00013
    }
    
    return costs

def calculate_storage_requirements(total_pairs):
    """Calculate storage requirements for different embedding dimensions"""
    storage = {}
    
    # Bytes per float32 number
    bytes_per_dimension = 4
    
    embedding_models = {
        'gemini_768d': 768,
        'gemini_3k_d': 3072,  # New Gemini model with 3K dimensions
        'openai_1536d': 1536,
        'sentence_transformers_384d': 384
    }
    
    for model, dimensions in embedding_models.items():
        embedding_size_mb = (total_pairs * dimensions * bytes_per_dimension) / (1024 * 1024)
        
        # Add metadata overhead (estimated 1KB per pair)
        metadata_size_mb = (total_pairs * 1024) / (1024 * 1024)
        
        storage[model] = {
            'dimensions': dimensions,
            'embedding_size_mb': embedding_size_mb,
            'metadata_size_mb': metadata_size_mb,
            'total_size_mb': embedding_size_mb + metadata_size_mb,
            'total_size_gb': (embedding_size_mb + metadata_size_mb) / 1024
        }
    
    return storage

def main():
    """Main analysis function"""
    processed_data_path = Path("data/processed/ai_cleaned")
    
    if not processed_data_path.exists():
        print(f"Error: {processed_data_path} does not exist")
        return
    
    print("🔍 Analyzing Ray Peat Corpus Scale...")
    print("=" * 60)
    
    # Analyze each major directory
    directories = [
        "01_Audio_Transcripts",
        "02_Publications", 
        "03_Chronological_Content",
        "04_Health_Topics",
        "05_Academic_Documents",
        "06_Email_Communications",
        "07_Special_Collections",
        "08_Newslatters"
    ]
    
    total_stats = {
        'total_files': 0,
        'total_pairs': 0,
        'total_tokens': 0,
        'directories': {}
    }
    
    for directory in directories:
        dir_path = processed_data_path / directory
        if dir_path.exists():
            print(f"\n📁 Analyzing {directory}...")
            dir_stats = analyze_directory(dir_path)
            total_stats['directories'][directory] = dir_stats
            total_stats['total_files'] += dir_stats['file_count']
            total_stats['total_pairs'] += dir_stats['total_pairs']
            total_stats['total_tokens'] += dir_stats['total_tokens']
            
            print(f"   Files: {dir_stats['file_count']}")
            print(f"   Q&A Pairs: {dir_stats['total_pairs']:,}")
            print(f"   Tokens: {dir_stats['total_tokens']:,}")
    
    # Overall summary
    print("\n" + "=" * 60)
    print("📊 CORPUS SCALE SUMMARY")
    print("=" * 60)
    print(f"Total Files Processed: {total_stats['total_files']:,}")
    print(f"Total Q&A Pairs: {total_stats['total_pairs']:,}")
    print(f"Total Tokens: {total_stats['total_tokens']:,}")
    print(f"Average Tokens per Pair: {total_stats['total_tokens'] / total_stats['total_pairs']:.1f}")
    print(f"Average Pairs per File: {total_stats['total_pairs'] / total_stats['total_files']:.1f}")
    
    # Calculate embedding costs
    costs = calculate_embedding_costs(total_stats['total_tokens'])
    print(f"\n💰 EMBEDDING COSTS")
    print("-" * 30)
    for provider, cost_info in costs.items():
        print(f"{provider.upper()}: ${cost_info['total_cost']:.2f}")
    
    # Calculate storage requirements
    storage = calculate_storage_requirements(total_stats['total_pairs'])
    print(f"\n💾 STORAGE REQUIREMENTS")
    print("-" * 30)
    for model, storage_info in storage.items():
        print(f"{model}: {storage_info['total_size_gb']:.2f} GB ({storage_info['dimensions']} dimensions)")
    
    # Detailed breakdown by directory
    print(f"\n📋 DETAILED BREAKDOWN BY CATEGORY")
    print("-" * 60)
    for dir_name, dir_stats in total_stats['directories'].items():
        if dir_stats['total_pairs'] > 0:
            print(f"\n{dir_name}:")
            print(f"  Files: {dir_stats['file_count']}")
            print(f"  Q&A Pairs: {dir_stats['total_pairs']:,}")
            print(f"  Tokens: {dir_stats['total_tokens']:,}")
            print(f"  Avg Tokens/Pair: {dir_stats['total_tokens'] / dir_stats['total_pairs']:.1f}")
    
    # Top 10 largest files
    all_files = []
    for dir_stats in total_stats['directories'].values():
        all_files.extend(dir_stats['files'])
    
    all_files.sort(key=lambda x: x['pair_count'], reverse=True)
    
    print(f"\n🏆 TOP 10 FILES BY Q&A PAIR COUNT")
    print("-" * 60)
    for i, file_info in enumerate(all_files[:10], 1):
        print(f"{i:2d}. {file_info['name']}")
        print(f"     Pairs: {file_info['pair_count']:,} | Tokens: {file_info['token_count']:,} | Size: {file_info['size_kb']:.1f}KB")
    
    # Save detailed results to JSON
    output_file = "corpus_scale_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': total_stats,
            'costs': costs,
            'storage': storage,
            'analysis_date': str(Path().resolve())
        }, f, indent=2)
    
    print(f"\n💾 Detailed analysis saved to: {output_file}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS")
    print("-" * 30)
    print(f"• Recommended Model: Gemini (768d) - ${costs['gemini']['total_cost']:.2f}, {storage['gemini_768d']['total_size_gb']:.1f}GB")
    print(f"• Vector Database: HNSW index for {total_stats['total_pairs']:,} vectors")
    print(f"• Expected Query Time: <100ms for similarity search")
    print(f"• RAM Requirements: ~{storage['gemini_768d']['total_size_gb'] * 1.5:.1f}GB for in-memory index")

if __name__ == "__main__":
    main() 