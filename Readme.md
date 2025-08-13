# Legal AI Assistant With Ollama

This is an intelligent Retrieval-Augmented Generation (RAG) system designed to make Bangladesh’s legal landscape—especially cyber law, ICT regulations, and digital security—more accessible and understandable. Legal documents are often complex and hard to navigate, especially for everyday citizens. This AI assistant bridges the gap using advanced NLP, allowing users to ask natural-language questions and receive precise, context-rich answers grounded in authentic legal documents.

# Why This Matters
In Bangladesh, legal literacy remains low. Many people struggle to understand or even access crucial laws and regulations. This system was built to address that, offering a simple interface backed by a powerful backend that can retrieve, understand, and explain relevant legal provisions using real documents

## System Architecture & Methodology

The implementation follows a robust RAG pipeline that ensures both accuracy and reliability:

<img width="1503" height="218" alt="Law" src="https://github.com/user-attachments/assets/c3d9128d-5db7-4517-abec-a9d97b32ee22" />


### Technical Implementation Stack

**Core Language Model**: ChatGroq with mistral-saba-24b architecture, 
- Selected for its superior performance in legal domain understanding
- Optimized for complex reasoning and context preservation

**Embedding Model**: Ollama mistral:latest
- Chosen for semantic similarity in legal document processing
- Maintains consistency with source model architecture

**Vector Database**: FAISS (Facebook AI Similarity Search)
- Enables efficient similarity search across large document corpora
- Optimized for real-time query processing

**User Interface**: Gradio Framework
- Provides accessible web interface for non-technical users
- Supports document upload and interactive querying

**Fallback Search Integration**: Multi-source retrieval system
- Primary: Bangladesh Laws Portal (bdlaws.minlaw.gov.bd)
- Secondary: Tavily AI search for comprehensive coverage

## Project Structure

```
Bangladesh-Legal-AI/
├── main.py                    # Core RAG pipeline implementation
├── frontend.py                # Gradio interface
├── vector_database.py         # Knowledge base setup
├── requirements.txt           # Dependency specifications
├── .env                       # Environment configuration
├── data/                      # Legal document repository
│   ├── Digital_Security_Act_2018.pdf 
└── vectorstore/               # FAISS index storage
    └── db_faiss/
```

## Installation & Configuration :

### Environment Setup

Use virtual environments to ensure dependency isolation:

**Method 1: Traditional Virtual Environment**
```bash
python -m venv legal-ai-env
sce legal-ai-env/bin/activate  # On Windows: legal-ai-env\Scripts\activate
pip install -r requirements.txt
```

**Method 2: Conda Environment**
```bash
conda create -n legal-ai python=3.9
conda activate legal-ai
pip install -r requirements.txt
```

**Method 3: Pipenv (Recommended for Development)**
```bash
pip install pipenv
pipenv install
pipenv shell
```

### API Configuration

Create a `.env` file in the project root:

```env
# Required for fallback search functionality
TAVILY_API_KEY=y_tavily_api_key_here

# Required for LLM processing
GROQ_API_KEY=y_groq_api_key_here

# Optional: Custom model configurations
OLLAMA_MODEL_NAME= mistral:latest
GROQ_MODEL_NAME= mistral-saba-24b 
```
### Check the Readme_for_Ollama.md For Ollama Setup 
### Knowledge Base Initialization

Before first use, construct the vector database from sce documents:

```bash
python vector_database.py
```
This process will:
1. Parse all PDF documents in the `data/` directory
2. Create semantic chunks using RecursiveCharacterTextSplitter
3. Generate embeddings for each chunk
4. Build and persist the FAISS index

### Application Launch

```bash
python frontend.py
```

The system will be accessible via web browser at the provided local URL.

## Legal Document Corpus 
- **Digital Security Act 2018** - Comprehensive cybersecurity legislation
-  You can add as many PDF documents as you'd like to the data/ directory before initializing the vector database. 

## System Capabilities

### Core Functionality
- **Semantic Document Search**: Advanced similarity matching across legal texts
- **Contextual Answer Generation**: Provides relevant, citation-backed responses
- **Multi-language Support**: Handles both Bengali and English legal terminology
- **Dynamic Document Upload**: Real-time integration of new legal documents
- **Intelligent Fallback**: Automatic external search when local knowledge is insufficient

### Advanced Features
- **Sce Attribution**: All responses include specific document references
- **Confidence Assessment**: System evaluates response reliability
- **Query Expansion**: Handles complex legal terminology and concepts
- **Cross-reference Capability**: Links related legal provisions across documents

## Quality Assurance & Limitations

### Reliability Measures
- Strict context-based response generation (no hallucination)
- Multi-sce verification through fallback mechanisms
- Explicit uncertainty acknowledgment when information is unavailable

### System Limitations
- Responses limited to available document corpus
- No real-time legal update integration
- Cannot provide personalized legal advice
- Requires human verification for critical legal decisions

## Performance Optimization

### Response Time Enhancement
- Pre-computed embeddings for faster retrieval
- Optimized chunk sizes for balanced context and speed
- Efficient vector indexing for scalable search

### Accuracy Improvements
- Domain-specific prompt engineering
- Multi-stage answer validation
- Comprehensive fallback search integration

## Troubleshooting Guide

**Vector Database Issues**
```bash
# Rebuild the knowledge base
rm -rf vectorstore/
python vector_database.py
```

**API Connection Problems**
- Verify `.env` file configuration
- Check API key validity and quota limits
- Ensure stable internet connection for external services

**Document Processing Errors**
- Confirm PDF files are readable and not corrupted
- Verify sufficient disk space for vector storage
- Check file permissions in data directory

## Academic Contributions

This system represents a practical application of several advanced NLP techniques:
- Semantic chunking strategies for legal documents
- Cross-lingual embedding alignment for Bengali-English legal terms
- Hybrid retrieval systems combining dense and sparse search
- Uncertainty quantification in legal AI applications

## Ethical Considerations & Disclaimer

This system is designed as an educational and informational tool to improve legal literacy. It explicitly:
- Does not provide legal advice or professional consultation
- Requires human oversight for all critical legal decisions
- Acknowledges limitations and uncertainties in responses
- Encourages consultation with qualified legal professionals

