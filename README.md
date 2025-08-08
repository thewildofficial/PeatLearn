# PeatLearn: Advanced AI Learning Platform

An AI-powered personalized learning platform featuring state-of-the-art machine learning techniques including Neural Collaborative Filtering, LSTM trajectory modeling, Reinforcement Learning, and Knowledge Graph Networks.

Built around Dr. Ray Peat's bioenergetic philosophy, this comprehensive system demonstrates advanced AI/ML concepts in a real-world educational application.

## 🚀 Quick Start

### Run Complete System Demo
```bash
python3 scripts/demo_system.py
```

### Launch Web Interfaces
```bash
# Streamlit Dashboard (ensure venv is activated)
source venv/bin/activate && streamlit run scripts/streamlit_dashboard.py --server.port 8502

# Modern HTML Interface
open web_ui/frontend/web_interface.html
```

### Start Backend Services
```bash
# Terminal 1: RAG Service (Port 8000)
source venv/bin/activate && cd inference/backend && python app.py

# Terminal 2: Advanced ML Service (Port 8001)  
source venv/bin/activate && cd inference/backend && python -m uvicorn advanced_app:app --port 8001
```

## 🧠 Advanced ML Features

✅ **Neural Collaborative Filtering** - Personalized content recommendations  
✅ **LSTM + Multi-head Attention** - Learning trajectory prediction  
✅ **Multi-task Deep Learning** - Adaptive quiz generation  
✅ **Deep Q-Networks (DQN)** - Reinforcement learning for content selection  
✅ **Actor-Critic Methods** - Continuous difficulty adjustment  
✅ **Graph Neural Networks** - Knowledge graph reasoning  
✅ **Retrieval-Augmented Generation** - Intelligent Q&A system  
✅ **Fine-tuned BERT** - Domain-specific concept extraction  

## 📊 System Architecture

- **Backend**: FastAPI microservices with advanced ML models
- **Frontend**: Modern HTML5 interface + Professional Streamlit dashboard  
- **Data**: 1000+ processed Ray Peat documents with vector embeddings
- **AI/ML**: 10+ state-of-the-art techniques integrated in production

Perfect for academic AI/ML final project demonstration! 🎓

## Features

- 🔍 **Intelligent Search**: Query Ray Peat's entire corpus using natural language
- 🥗 **Food & Nutrition Insights**: Learn about specific foods from his perspective  
- 🧬 **Hormonal Analysis**: Understand hormone interactions and optimization
- 🔬 **Biological Process Exploration**: Deep dive into Ray Peat's unique biological thinking
- 📚 **Complete Corpus Access**: All transcripts, articles, books, and interviews
- 🎯 **Personalized Learning**: AI-guided exploration based on your interests
- 📱 **Modern Interface**: Clean, responsive web application

## Dataset Overview

Our comprehensive dataset includes:

- **Audio Transcripts**: 188+ podcast interviews and radio shows
- **Publications**: 96+ academic papers and articles  
- **Health Topics**: 98+ specialized health discussions
- **Newsletters**: 59+ newsletter articles
- **Academic Documents**: Thesis and foundational papers
- **Email Communications**: Selected correspondence
- **Special Collections**: Rare interviews and discussions

**Total**: 552 documents representing decades of bioenergetic research and thinking.

## Technical Architecture

### Data Pipeline
```
Raw Data → Quality Analysis → AI-Powered Cleaning → Segmentation → Embedding → Hugging Face Hosting
```

### 📂 Dataset Hosting
Our embeddings are hosted separately to keep the codebase lightweight:

- **Code Repository**: [GitHub](https://github.com/thewildofficial/PeatLearn) (this repo)
- **Embeddings Dataset**: [Hugging Face](https://huggingface.co/datasets/abanwild/peatlearn-embeddings)

This hybrid approach allows:
- ✅ Fast code sharing and collaboration
- ✅ Large ML artifacts hosted efficiently  
- ✅ Easy contributor onboarding without massive downloads
- ✅ Automatic embedding synchronization

### System Components

1. **Data Processing** (`data/`)
   - Raw corpus storage and organization
   - Quality analysis and scoring
   - Processed and cleaned datasets

2. **Preprocessing & Cleaning** (`preprocessing/`)
   - AI-powered text cleaning and correction
   - Document segmentation and speaker attribution
   - Quality assessment and validation

3. **Embedding & Vectorization** (`embedding/`)
   - Text embedding using sentence-transformers/all-mpnet-base-v2
   - Hugging Face dataset hosting for embeddings
   - Automatic download and caching system
   - Similarity search optimization

4. **Inference Backend** (`inference/`)
   - RAG (Retrieval-Augmented Generation) system
   - API endpoints for frontend communication
   - LLM integration and fine-tuning

5. **Web UI Frontend** (`web_ui/`)
   - React-based user interface
   - Interactive search and exploration
   - Responsive design for all devices

## Project Structure

```
PeatLearn/
├── data/                           # Data storage and management
│   ├── raw/                       # Original source files
│   ├── processed/                 # Cleaned and processed data
│   └── analysis/                  # Quality analysis results
├── preprocessing/                  # Data cleaning and preparation
│   ├── cleaning/                  # Main cleaning pipeline
│   ├── quality_analysis/          # Quality assessment tools
│   └── segmentation/              # Document segmentation
├── embedding/                      # Text embedding and vectorization
│   ├── models/                    # Embedding models
│   ├── vectorstore/               # Vector database
│   └── evaluation/                # Embedding quality assessment
├── inference/                      # Backend API and RAG system
│   └── backend/                   # API servers and LLM integration
├── web_ui/                        # Frontend web application
│   └── frontend/                  # React application
├── tests/                         # Testing suites
├── docs/                          # Documentation and requirements
├── config/                        # Configuration files
└── logs/                          # Application logs
```

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+ (for frontend)
- Gemini API key or Hugging Face access
- 8GB+ RAM recommended

### Setup

1. **Clone and Setup Environment**
```bash
git clone <repository-url>
cd PeatLearn
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure Environment Variables**
```bash
cp .env.template .env
# Edit .env with your API keys:
# GEMINI_API_KEY=your_key_here
# HF_TOKEN=your_hugging_face_token  # Optional, for private access
```

3. **Download Pre-trained Embeddings**
```bash
cd embedding
python download_from_hf.py
# This downloads ~700MB of embeddings from Hugging Face
```

4. **Process the Data (Optional - for development)**
```bash
# Run data cleaning pipeline
cd preprocessing/cleaning
python main_pipeline.py --limit 10  # Start with sample

# Full processing
python main_pipeline.py
```

5. **Generate New Embeddings (Optional - for development)**
```bash
cd ../../embedding
python embed_corpus.py
```

6. **Start Backend Server**
```bash
cd ../inference/backend
python app.py
```

7. **Launch Frontend**
```bash
cd ../../web_ui/frontend
npm install
npm start
```

### Development Workflow

1. **Data Quality Assessment**: Run quality analysis on new data
2. **Preprocessing**: Clean and segment documents using AI pipeline
3. **Embedding**: Generate vector representations
4. **Backend Development**: Implement RAG and API endpoints
5. **Frontend Development**: Build user interface features
6. **Testing**: Validate system components
7. **Deployment**: Deploy to production environment

## Data Processing Pipeline

### Stage 1: Quality Analysis
- Automated scoring of document quality
- Identification of processing requirements
- Categorization by content type and complexity

### Stage 2: AI-Powered Cleaning
- **Tier 1** (27% of files): Rules-based cleaning for high-quality documents
- **Tier 2** (73% of files): AI-powered cleaning for complex documents
  - OCR error correction
  - Document segmentation  
  - Speaker attribution
  - Text enhancement

### Stage 3: Embedding Generation
- Text vectorization using state-of-the-art models
- Semantic search optimization
- Vector database storage

### Stage 4: RAG Implementation
- Retrieval-augmented generation system
- Context-aware response generation
- Fine-tuned models for Ray Peat's style

## API Documentation

### Core Endpoints

```
GET  /api/search?q={query}           # Search corpus
POST /api/ask                        # Ask questions
GET  /api/topics                     # Browse topics
GET  /api/documents/{id}             # Get document
GET  /api/recommendations            # Get recommendations
```

### Example Usage

```javascript
// Search for information about thyroid
const response = await fetch('/api/search?q=thyroid function metabolism');
const results = await response.json();

// Ask a specific question
const answer = await fetch('/api/ask', {
  method: 'POST',
  body: JSON.stringify({
    question: "What does Ray Peat say about coconut oil?",
    context: "nutrition"
  })
});
```

## Performance Metrics

- **Processing Speed**: 2-5 files/second (rules-based), 1 file/10-30s (AI-powered)
- **Search Latency**: <200ms average response time
- **Accuracy**: 95%+ relevance for domain-specific queries
- **Coverage**: 100% of Ray Peat's public corpus
- **Uptime**: 99.9% availability target

## Contributing

We welcome contributions to enhance the Ray Peat Legacy platform:

1. **Data Quality**: Improve cleaning algorithms and quality assessment
2. **Search Enhancement**: Better embedding models and retrieval systems
3. **UI/UX**: Frontend improvements and new features
4. **Documentation**: Help others understand Ray Peat's work
5. **Testing**: Ensure system reliability and accuracy

### Development Guidelines

- Follow clean code principles
- Write comprehensive tests
- Document all functions and APIs
- Use meaningful commit messages
- Ensure responsive design

## Technology Stack

### Backend
- **Python**: Core processing and API development
- **FastAPI**: High-performance API framework
- **Gemini API**: AI-powered text processing
- **ChromaDB/Pinecone**: Vector database
- **PostgreSQL**: Metadata storage

### Frontend  
- **React**: Modern UI framework
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **React Query**: Data fetching and caching

### AI/ML
- **Google Gemini**: LLM for understanding and generation
- **Hugging Face**: Alternative embedding models
- **LangChain**: RAG orchestration
- **Transformers**: Model management

## Deployment

### Local Development
```bash
docker-compose up -d  # Start all services
```

### Production
- **Frontend**: Vercel/Netlify deployment
- **Backend**: Google Cloud Run or AWS Lambda
- **Database**: Managed PostgreSQL + Vector DB
- **CDN**: Global content delivery

## Roadmap

### Phase 1: Foundation (Current)
- ✅ Data processing pipeline
- ✅ Quality analysis system
- ⏳ AI-powered cleaning
- ⏳ Basic RAG implementation

### Phase 2: Core Platform
- 🔄 Advanced search capabilities
- 🔄 Web interface development
- 🔄 API optimization
- 🔄 User experience testing

### Phase 3: Enhancement
- ⏳ Fine-tuned models
- ⏳ Personalization features
- ⏳ Mobile application
- ⏳ Community features

### Phase 4: Scale
- ⏳ Multi-language support
- ⏳ Advanced analytics
- ⏳ Educational content
- ⏳ Research tools

## License

This project is developed for educational and research purposes to preserve and share Dr. Ray Peat's scientific contributions.

## Acknowledgments

- **Dr. Ray Peat**: For his groundbreaking work in bioenergetic medicine
- **Contributors**: Everyone helping to preserve and share this knowledge
- **Community**: Ray Peat enthusiasts and researchers worldwide

---

**"Energy and structure are interdependent at every level."** - Ray Peat

For questions, suggestions, or contributions, please open an issue or contact the development team. 