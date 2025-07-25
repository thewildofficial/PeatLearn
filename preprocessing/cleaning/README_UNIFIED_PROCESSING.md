# Unified Ray Peat Signal Processing System

This directory contains a streamlined, unified processing system optimized for **million-token context windows** and maximum Ray Peat signal extraction.

## 🚀 Quick Start

### Process 10 files with rules-based cleaning (no API key required):
```bash
cd preprocessing/cleaning
python run_unified_processing.py \
  --input-dir ../../data/raw/raw_data \
  --output-dir ../../data/processed/signal_extracted \
  --limit 10
```

### Process with AI enhancement (requires Google API key):
```bash
cd preprocessing/cleaning
python run_unified_processing.py \
  --input-dir ../../data/raw/raw_data \
  --output-dir ../../data/processed/signal_extracted \
  --analysis-file ../../data/analysis/corpus_analysis.csv \
  --api-key YOUR_GOOGLE_API_KEY
```

## 📁 System Components

### Core Modules

1. **`unified_signal_processor.py`** - Main processing engine
   - Rules-based cleaning for all files
   - AI enhancement for low-signal content
   - Intelligent quality assessment
   - Unified processing strategy

2. **`mega_chunker.py`** - Advanced chunking system
   - Creates massive chunks (up to 900K characters ≈ 225K tokens)
   - Preserves conversation boundaries and context
   - Optimized for million-token LLMs
   - Ray Peat signal density analysis

3. **`run_unified_processing.py`** - Execution script
   - Complete pipeline orchestration
   - Progress tracking and logging
   - Flexible configuration options

### Supporting Modules

4. **`rules_based_cleaners.py`** - Basic cleaning functions
   - HTML content extraction
   - Whitespace normalization
   - Artifact removal

5. **`ai_powered_cleaners.py`** - AI enhancement functions
   - Advanced signal extraction
   - Speaker attribution
   - Noise removal

## 🎯 Processing Strategy

### Phase 1: Signal Extraction
1. **Rules-based cleaning** applied to all files
2. **Signal quality assessment** using Ray Peat content patterns
3. **AI enhancement** for files below signal threshold (if API available)
4. **Best result selection** based on signal quality improvement

### Phase 2: Mega-Chunking
1. **Intelligent chunking** that preserves:
   - Complete conversations and speaker turns
   - Topic boundaries and section headers
   - Educational context and flow
2. **Massive chunks** optimized for million-token context windows
3. **Metadata enrichment** with signal density and topic analysis

## 📊 Key Features

### Million-Token Optimization
- **Massive chunks**: Up to 900K characters (≈225K tokens)
- **Context preservation**: Intelligent break point detection
- **Overlap management**: Configurable overlap between chunks

### Signal Quality Focus
- **Ray Peat detection**: Keyword-based content analysis
- **Speaker attribution**: Clear **RAY PEAT:** vs **HOST:** labeling
- **Educational value**: Preserves complete explanations and insights

### Flexible Processing
- **Tier-aware**: Uses quality analysis for smarter processing decisions
- **Fallback strategy**: Rules-only if AI unavailable or fails
- **Configurable limits**: Test with small batches, scale to full corpus

## 🗂️ Output Structure

```
data/processed/signal_extracted/
├── logs/                           # Processing logs
├── mega_chunks/                    # Chunked content
│   ├── file1/
│   │   ├── file1_chunk_000.txt    # Individual chunks
│   │   ├── file1_chunk_001.txt
│   │   └── file1_chunks_metadata.json
│   └── chunking_summary.json      # Overall chunking stats
├── file1_processed.txt             # Processed files
├── file2_processed.txt
├── processing_metadata.json        # Processing details
└── pipeline_summary.json          # Complete pipeline summary
```

## ⚙️ Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input-dir` | Raw files directory | Required |
| `--output-dir` | Output directory | Required |
| `--analysis-file` | Quality analysis CSV | Optional |
| `--limit` | File processing limit | No limit |
| `--api-key` | Google API key | From env var |
| `--no-chunking` | Skip mega-chunking | False |
| `--max-chunk-size` | Max chunk characters | 900,000 |
| `--verbose` | Detailed logging | False |

## 🔍 Quality Metrics

### Signal Assessment
- **Ray Peat density**: Ratio of Ray Peat content to total content
- **Topic coverage**: Number of bioenergetic topics identified
- **Speaker attribution**: Quality of conversation labeling

### Processing Efficiency
- **Rules vs AI**: Percentage using each method
- **Signal improvement**: AI enhancement effectiveness
- **Compression ratio**: Output size vs input size

## 📈 Example Results

```bash
📊 FINAL METRICS:
  ✅ Files processed: 25
  🔧 Rules-only: 18
  🤖 AI-enhanced: 7
  📈 Avg signal ratio: 0.742
  📦 Mega-chunks created: 89
  🎯 Avg chunk density: 0.816
  🔤 Total tokens: 1,847,293
```

## 🚨 Troubleshooting

### Common Issues

1. **No API key**: System falls back to rules-only processing
2. **Large files**: Automatically chunked for AI processing
3. **Low signal files**: Enhanced with AI if available
4. **Memory issues**: Reduce `--max-chunk-size` parameter

### Debug Mode
```bash
python run_unified_processing.py --verbose [other args]
```

## 🔄 Migration from Old System

### Archived Scripts
The following scripts have been moved to `archive/`:
- `main_pipeline.py` (replaced by `unified_signal_processor.py`)
- `process_tier1_only.py` (functionality integrated)
- `extract_ray_peat_signal.py` (improved version integrated)

### Key Improvements
- **Unified approach**: Single script handles all processing
- **Million-token chunks**: Much larger context windows
- **Better signal detection**: Enhanced keyword matching
- **Smarter AI usage**: Only when needed, better prompts
- **Complete pipeline**: Processing + chunking in one run

## 📋 Next Steps

1. **Test with small batch**: `--limit 5` to verify functionality
2. **Scale to full corpus**: Remove limit for complete processing
3. **Optimize chunk size**: Adjust for your specific model context window
4. **Monitor quality**: Review signal ratios and chunk densities

This unified system maximizes Ray Peat signal extraction while leveraging your million-token context window for superior educational content creation. 