# Pinecone Migration for Ray Peat Embeddings

This directory contains all the tools and scripts needed to migrate your Ray Peat corpus embeddings to Pinecone vector database.

## 📋 Overview

The Pinecone migration provides:
- **Scalable vector storage** with Pinecone's managed infrastructure
- **Enhanced search capabilities** with metadata filtering
- **Improved performance** for production workloads
- **Better reliability** and availability
- **Advanced features** like namespaces and hybrid search

## 🚀 Quick Start

### Prerequisites

1. **Pinecone API Key**: Get your free API key from [Pinecone](https://www.pinecone.io/)
2. **Existing embeddings**: Make sure you have generated embeddings using `embed_corpus.py`
3. **Environment setup**: Add your API key to `.env` file

### 1. Install Dependencies

```bash
cd embedding/pinecone
pip install -r requirements.txt
```

### 2. Set Up Environment

Add your Pinecone API key to the project's `.env` file:

```bash
# In /Users/aban/drive/Projects/PeatLearn/.env
PINECONE_API_KEY=your_pinecone_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here  # Still needed for query embeddings
```

### 3. Upload Existing Embeddings

```bash
python upload.py
```

This will:
- Create a new Pinecone index called `ray-peat-corpus`
- Upload all your existing embeddings and metadata
- Verify the upload was successful
- Generate a detailed upload report

### 4. Test the Migration

```bash
python test_pinecone.py
```

This runs comprehensive tests to ensure everything is working correctly.

## 📁 File Structure

```
embedding/pinecone/
├── __init__.py              # Package initialization
├── README.md               # This documentation
├── requirements.txt        # Pinecone-specific dependencies
├── upload.py              # Main migration script
├── vector_search.py       # Pinecone vector search engine
├── rag_system.py         # Pinecone-based RAG system
├── utils.py              # Management and utility functions
└── test_pinecone.py      # Comprehensive test suite
```

## 🔧 Configuration

### Pinecone Index Settings

Default configuration (can be modified in `upload.py`):

```python
PINECONE_API_KEY = "your_api_key"       # From environment
INDEX_NAME = "ray-peat-corpus"          # Index name
VECTOR_DIMENSION = 768                  # Gemini embedding dimensions
BATCH_SIZE = 128                        # Upload batch size
METRIC = "cosine"                       # Similarity metric
CLOUD = "aws"                          # Cloud provider
REGION = "us-east-1"                   # Region
```

### Metadata Structure

Each vector includes the following metadata:

```json
{
  "context": "Question or conversation context (max 1000 chars)",
  "ray_peat_response": "Ray Peat's response (up to 40KB)",
  "source_file": "Original source file name",
  "tokens": 139,
  "original_id": "document_1_processed_5",
  "truncated": false
}
```

## 🔍 Usage Examples

### Vector Search

```python
from embedding.pinecone.vector_search import PineconeVectorSearch

# Initialize search engine
search_engine = PineconeVectorSearch()

# Search for relevant passages
results = await search_engine.search(
    query="thyroid hormone metabolism",
    top_k=10,
    min_similarity=0.3
)

# Search with metadata filters
filtered_results = await search_engine.search(
    query="energy production",
    top_k=5,
    filter_dict={"tokens": {"$gte": 100}}  # Only longer responses
)
```

### RAG System

```python
from embedding.pinecone.rag_system import PineconeRAG

# Initialize RAG system
rag = PineconeRAG()

# Ask a question
response = await rag.answer_question(
    question="What did Ray Peat say about sugar and metabolism?",
    max_sources=5,
    min_similarity=0.3
)

print(f"Answer: {response.answer}")
print(f"Sources: {len(response.sources)}")
print(f"Confidence: {response.confidence}")
```

### Management Utilities

```python
from embedding.pinecone.utils import PineconeManager

# Initialize manager
manager = PineconeManager()

# Get index statistics
stats = manager.get_index_info()
print(f"Total vectors: {stats['total_vector_count']}")

# Verify data integrity
verification = manager.verify_vector_integrity(sample_size=100)
print(f"Health score: {verification.get('health_score', 'N/A')}")

# Backup metadata
manager.backup_index_metadata()
```

## 🎯 Advanced Features

### Metadata Filtering

Pinecone supports powerful metadata filtering:

```python
# Filter by source file
filter_dict = {"source_file": "specific_document.txt"}

# Filter by token count range
filter_dict = {"tokens": {"$gte": 50, "$lte": 500}}

# Multiple filters
filter_dict = {
    "source_file": {"$in": ["doc1.txt", "doc2.txt"]},
    "tokens": {"$gte": 100}
}
```

### Source-Specific Search

```python
# Search only within specific documents
response = await rag.answer_with_source_filter(
    question="What about thyroid function?",
    source_files=["hormones_processed.txt", "metabolism_processed.txt"]
)
```

### Similar Document Discovery

```python
# Find documents similar to a specific one
similar = await search_engine.get_similar_documents(
    document_id="document_123",
    top_k=10
)
```

## 📊 Performance Considerations

### Optimization Tips

1. **Batch Size**: Adjust `BATCH_SIZE` in upload.py based on your network
2. **Top-K Limits**: Use reasonable top_k values (10-50) for better performance
3. **Metadata Filtering**: Use filters to reduce search space
4. **Similarity Thresholds**: Set appropriate min_similarity to filter irrelevant results

### Expected Performance

- **Search Latency**: ~100-300ms for typical queries
- **RAG Response**: ~2-5 seconds (including LLM generation)
- **Upload Speed**: ~1000-2000 vectors per minute
- **Storage Efficiency**: ~40KB per vector (including metadata)

## 🛠️ Management Operations

### Using the Interactive Utility

```bash
python utils.py
```

This provides an interactive menu for:
- Viewing index information
- Verifying data integrity
- Sampling vectors for inspection
- Managing source files
- Creating backups
- Generating health reports

### Command Line Operations

```python
# Get index stats
python -c "
from embedding.pinecone.utils import PineconeManager
manager = PineconeManager()
print(manager.get_index_info())
"

# Run health check
python -c "
from embedding.pinecone.utils import PineconeManager
import json
manager = PineconeManager()
report = manager.generate_health_report()
print(json.dumps(report, indent=2))
"
```

## 🔄 Migration from Local Storage

### Updating Existing Code

If you're migrating from the local vector search, update your imports:

```python
# Old (local storage)
from inference.backend.rag.vector_search import RayPeatVectorSearch
from inference.backend.rag.rag_system import RayPeatRAG

# New (Pinecone)
from embedding.pinecone.vector_search import PineconeVectorSearch
from embedding.pinecone.rag_system import PineconeRAG
```

### API Compatibility

The Pinecone implementation maintains API compatibility with the original:

- `search()` method works the same way
- `SearchResult` objects have the same structure
- `RAGResponse` objects are identical
- Global instances (`search_engine`, `rag_system`) are available

## 🚨 Troubleshooting

### Common Issues

**1. "PINECONE_API_KEY not found"**
- Add your API key to the `.env` file in the project root
- Make sure the `.env` file is in `/Users/aban/drive/Projects/PeatLearn/`

**2. "Index not found"**
- Run `upload.py` first to create and populate the index
- Check that upload completed successfully

**3. "No vectors found in index"**
- Verify upload completed: check the upload report
- Use `utils.py` to inspect index contents

**4. "Import errors"**
- Install dependencies: `pip install -r requirements.txt`
- Make sure you're in the right directory

**5. "Rate limit errors during upload"**
- The script includes automatic retry logic
- For persistent issues, reduce `BATCH_SIZE` in upload.py

### Getting Help

1. **Check upload reports**: Look at generated JSON reports for errors
2. **Run test suite**: `python test_pinecone.py` for comprehensive diagnostics
3. **Use utilities**: `python utils.py` for interactive troubleshooting
4. **Check logs**: The scripts provide detailed logging output

## 💰 Cost Considerations

### Pinecone Pricing (as of 2024)

- **Free Tier**: 1 index, 100K vectors, sufficient for testing
- **Starter**: $70/month for production use
- **Standard**: $280/month for larger deployments

### Ray Peat Corpus Estimates

- **Vectors**: ~26,000 (fits in free tier for development)
- **Storage**: ~100MB metadata + vectors
- **Monthly Cost**: Free tier for development, $70+ for production

### Cost Optimization

1. **Use free tier** for development and testing
2. **Delete unused indexes** to save costs
3. **Monitor usage** through Pinecone dashboard
4. **Consider alternatives** like Qdrant or Weaviate for cost-sensitive deployments

## 🔮 Future Enhancements

### Planned Features

1. **Namespace support** for multi-tenant deployments
2. **Hybrid search** combining vector and keyword search
3. **Real-time updates** for new content
4. **Advanced analytics** and usage tracking
5. **Integration** with other Pinecone features

### Contributing

To contribute improvements:

1. Test your changes with `test_pinecone.py`
2. Update this README if adding new features
3. Follow the existing code style and patterns
4. Add appropriate error handling and logging

## 📚 Additional Resources

- [Pinecone Documentation](https://docs.pinecone.io/)
- [Pinecone Python Client](https://github.com/pinecone-io/pinecone-python-client)
- [Vector Database Best Practices](https://www.pinecone.io/learn/)
- [Ray Peat Archive](https://raypeat.com/) - Original source material
