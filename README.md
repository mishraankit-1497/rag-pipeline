# RAG Pipeline - Retrieval-Augmented Generation

A comprehensive Retrieval-Augmented Generation (RAG) pipeline that enables intelligent document retrieval and question-answering using embeddings and vector search. This project demonstrates a complete workflow for building RAG systems with LangChain, ChromaDB, and SentenceTransformers.

## 🎯 Overview

This RAG pipeline implements a complete end-to-end workflow for:
- **Document Loading**: Load documents from multiple sources (PDF, TXT files)
- **Document Chunking**: Split large documents into manageable chunks with overlap
- **Embedding Generation**: Generate semantic embeddings using pre-trained models
- **Vector Storage**: Store embeddings in a persistent vector database (ChromaDB)
- **Semantic Search**: Retrieve relevant documents based on query similarity
- **Question Answering**: Answer user queries using retrieved documents

## ✨ Features

- **Multi-Source Document Loading**: Support for PDF and text file formats
- **Intelligent Chunking**: RecursiveCharacterTextSplitter for context-aware chunking
- **Semantic Embeddings**: Uses SentenceTransformer's "all-MiniLM-L6-v2" model
- **Persistent Vector Storage**: ChromaDB for scalable vector storage
- **Similarity-Based Retrieval**: Find relevant documents with configurable thresholds
- **Scored Results**: Retrieve documents with relevance scores
- **Error Handling**: Comprehensive error handling and logging

## 📁 Project Structure

```
rag-pipeline/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
├── main.py                  # Entry point
├── notebook/
│   └── document.ipynb       # Complete RAG implementation notebook
├── data/
│   ├── text_files/          # Text documents
│   ├── pdf/                 # PDF documents
│   └── chroma_db/           # Vector database storage
└── .gitignore
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/mishraankit-1497/rag-pipeline.git
cd rag-pipeline
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Notebook

Open and run the Jupyter notebook to execute the complete RAG pipeline:

```bash
jupyter notebook notebook/document.ipynb
```

### Using the RAG Pipeline Programmatically

```python
from notebook.document import RAGRetriever, VectorStore, EmbeddingManager

# Initialize components
embedding_manager = EmbeddingManager()
vector_store = VectorStore()
rag_retriever = RAGRetriever(vector_store, embedding_manager)

# Retrieve relevant documents
query = "What is the introduction to LLMs?"
results = rag_retriever.retrieve(query, top_k=5, score_threshold=0.0)

# Process results
for doc in results:
    print(f"Score: {doc['score']:.4f}")
    print(f"Content: {doc['content']}")
    print(f"Metadata: {doc['metadata']}")
    print("---")
```

## 🏗️ Architecture

### Pipeline Flow

```
Documents (PDF/TXT) 
    ↓
Document Loader (DirectoryLoader, TextLoader, PyMuPDFLoader)
    ↓
Document Chunker (RecursiveCharacterTextSplitter)
    ↓
Embedding Generator (SentenceTransformer)
    ↓
Vector Store (ChromaDB)
    ↓
Retriever (Similarity Search)
    ↓
Ranked Results with Scores
```

## 🔧 Components

### 1. **DocumentLoaders**
- `TextLoader`: Loads text files with specified encoding
- `DirectoryLoader`: Recursively loads all files matching a pattern
- `PyMuPDFLoader`: Loads PDF documents

### 2. **EmbeddingManager**
Handles semantic embedding generation using SentenceTransformer:
- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Generates embeddings for text chunks
- Supports batch processing with progress tracking

```python
class EmbeddingManager:
    def generate_embeddings(texts: List[str]) -> np.ndarray
```

### 3. **VectorStore**
Manages document storage using ChromaDB:
- Creates persistent collections
- Stores embeddings and document metadata
- Supports efficient similarity search

```python
class VectorStore:
    def add_documents(documents: List[Any], embeddings: np.ndarray)
```

### 4. **RAGRetriever**
Implements semantic search and retrieval:
- Generates query embeddings
- Searches vector store using similarity metrics
- Filters results by score threshold
- Returns ranked documents with relevance scores

```python
class RAGRetriever:
    def retrieve(query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict]
```

## ⚙️ Configuration

### Document Chunking
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,           # Characters per chunk
    chunk_overlap=200,         # Overlap between chunks
    length_function=len
)
```

### Embedding Model
- **Model Name**: `all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Source**: Hugging Face Sentence Transformers

### Vector Store
- **Database**: ChromaDB
- **Storage**: `./chroma_db/` (persistent)
- **Collection**: "documents"

### Retrieval Parameters
- `top_k`: Number of documents to retrieve (default: 5)
- `score_threshold`: Minimum relevance score (default: 0.0)

## 📊 Examples

### Example 1: Load and Chunk Documents

```python
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

loader = DirectoryLoader(
    "../data/pdf",
    glob="*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=False
)
documents = loader.load()
```

### Example 2: Generate Embeddings

```python
texts = [doc.page_content for doc in chunks]
embeddings = embedding_manager.generate_embeddings(texts)
vector_store.add_documents(documents=texts, embeddings=embeddings)
```

### Example 3: Retrieve Relevant Documents

```python
results = rag_retriever.retrieve(
    query="What is the introduction to LLMs?",
    top_k=5,
    score_threshold=0.5
)

for result in results:
    print(f"ID: {result['id']}")
    print(f"Content: {result['content']}")
    print(f"Relevance Score: {result['score']:.4f}")
    print(f"Rank: {result['rank']}")
```

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | LLM framework and tools |
| `langchain-core` | Core LangChain abstractions |
| `langchain-community` | Community document loaders |
| `sentence-transformers` | Semantic embeddings |
| `chromadb` | Vector database |
| `pypdf` | PDF handling |
| `pymupdf` | PDF extraction (alternative) |
| `faiss-cpu` | Similarity search |
| `langchain-groq` | Groq LLM integration |

## 🔄 Workflow

1. **Data Preparation**: Place your documents in `data/text_files/` or `data/pdf/`
2. **Document Loading**: Load documents using appropriate loaders
3. **Chunking**: Split documents into overlapping chunks
4. **Embedding**: Generate semantic embeddings for each chunk
5. **Indexing**: Store embeddings in ChromaDB
6. **Query**: Ask questions to retrieve relevant documents
7. **Ranking**: Results are ranked by similarity score

