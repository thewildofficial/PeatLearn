# Ray Peat Legacy - Embedding Module

This module handles the conversion of Ray Peat's processed Q&A pairs into vector embeddings using Google's Gemini API, enabling semantic search capabilities.

## 📊 Pipeline Overview

```
Processed Q&A Pairs → Gemini Embeddings → Vector Storage → RAG System
     (26K pairs)         (768D vectors)     (100MB)      (Search Ready)
```

## 🎯 What This Module Does

### Input
- **26,326 Q&A pairs** from the AI-cleaned corpus
- Each pair contains:
  - **Context**: Question or conversation trigger
  - **Ray Peat Response**: His detailed answer
  - **Source**: Original document reference

### Process
1. **Parse Q&A pairs** from processed text files
2. **Generate embeddings** using Gemini `gemini-embedding-exp-03-07` model
3. **Store vectors** in multiple formats (pickle, numpy)
4. **Track progress** and costs in real-time
5. **Generate reports** for analysis

### Output
- **Vector embeddings** (768-dimensional)
- **Metadata files** with Q&A content
- **Cost reports** and progress tracking
- **Multiple storage formats** for flexibility

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Set up environment and dependencies
python embedding/setup_env.py

# Or manually create .env file:
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 2. Install Dependencies
```bash
pip install -r embedding/requirements.txt
```

### 3. Generate Embeddings
```bash
python embedding/embed_corpus.py
```

## 📁 File Structure

```
embedding/
├── embed_corpus.py      # Main embedding pipeline
├── setup_env.py         # Environment setup helper
├── requirements.txt     # Embedding dependencies
├── README.md           # This documentation
├── vectors/            # Generated embeddings output
│   ├── ray_peat_embeddings_YYYYMMDD_HHMMSS.pkl
│   ├── embeddings_YYYYMMDD_HHMMSS.npy
│   ├── metadata_YYYYMMDD_HHMMSS.json
│   └── embedding_report.json
├── models/             # Future model storage
└── vectorstore/        # Vector database files
    └── chroma/         # ChromaDB persistence
```

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key (required)
- `GEMINI_RATE_LIMIT`: Requests per minute (default: 15)
- `EMBEDDING_MODEL`: Model name (default: gemini-embedding-exp-03-07)
- `EMBEDDING_DIMENSIONS`: Vector dimensions (default: 768)

### Settings (config/settings.py)
```python
EMBEDDING_MODEL = "gemini-embedding-exp-03-07"
EMBEDDING_DIMENSIONS = 768
GEMINI_RATE_LIMIT = 15  # requests per minute
```

## 📊 Expected Results

Based on our corpus analysis:

### Scale
- **Total Q&A Pairs**: 26,326
- **Total Tokens**: 3,660,641
- **Average Tokens per Pair**: 139.1

### Cost Estimation
- **Gemini Embedding Cost**: ~$0.37 total
- **Processing Time**: ~30 minutes (at 15 req/min)
- **Storage Required**: ~100MB (768D vectors)

### Output Files
- **`ray_peat_embeddings_*.pkl`**: Complete Q&A pairs with embeddings
- **`embeddings_*.npy`**: Just the vector arrays (26,326 x 768)
- **`metadata_*.json`**: Q&A content and source information
- **`embedding_report.json`**: Processing statistics and costs

## 🎛️ Advanced Usage

### Custom Batch Processing
```python
from embedding.embed_corpus import RayPeatCorpusEmbedder

embedder = RayPeatCorpusEmbedder()
qa_pairs = embedder.load_all_qa_pairs()

# Generate embeddings in smaller batches
embeddings = await embedder.generate_embeddings_batch(qa_pairs, batch_size=5)
```

### Loading Generated Embeddings
```python
import pickle
import numpy as np

# Load complete Q&A pairs with embeddings
with open('embedding/vectors/ray_peat_embeddings_20241220_143022.pkl', 'rb') as f:
    qa_pairs = pickle.load(f)

# Or load just vectors and metadata
embeddings = np.load('embedding/vectors/embeddings_20241220_143022.npy')
with open('embedding/vectors/metadata_20241220_143022.json', 'r') as f:
    metadata = json.load(f)
```

## 🔍 Embedding Quality

### Vector Dimensions
- **768D vectors** from Gemini gemini-embedding-exp-03-07
- **Normalized embeddings** for cosine similarity
- **High-quality representations** for semantic search

### Content Processing
- **Combined context + response** for comprehensive embedding
- **Preserves semantic relationships** between Q&A pairs
- **Maintains source attribution** for result transparency

## 🚦 Rate Limiting & Costs

### API Limits
- **15 requests per minute** (Gemini free tier)
- **Automatic rate limiting** built into pipeline
- **Retry logic** for failed requests

### Cost Tracking
- **Real-time cost calculation** during processing
- **Token-based pricing** ($0.00001 per 1K tokens)
- **Final cost report** with detailed breakdown

## 🔗 Next Steps

After embedding generation, you can:

1. **Set up vector database** (ChromaDB, Pinecone, etc.)
2. **Build RAG system** for semantic search
3. **Create web interface** for user queries
4. **Implement similarity search** for related Q&As

## 🐛 Troubleshooting

### Common Issues

**API Key Not Found**
```bash
# Add to .env file
echo "GEMINI_API_KEY=your_key" > .env
```

**Rate Limit Exceeded**
- Reduce `GEMINI_RATE_LIMIT` in settings
- Check your Gemini API quota

**Memory Issues**
- Process in smaller batches
- Use numpy format instead of pickle

**Missing Dependencies**
```bash
pip install -r embedding/requirements.txt
```

## 📈 Performance Monitoring

The pipeline provides detailed progress tracking:
- **Real-time progress** with ETA estimates
- **Success/failure rates** for API calls
- **Cost tracking** throughout processing
- **Detailed reports** saved to JSON

## 🔜 Future Enhancements

- **Vector database integration** (ChromaDB, Pinecone)
- **Chunk optimization** for better retrieval
- **Multiple embedding models** comparison
- **Incremental updates** for new content
- **Semantic clustering** analysis 