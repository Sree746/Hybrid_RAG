# Hybrid RAG System

A Retrieval-Augmented Generation (RAG) system that uses hybrid search and reranking to generate answers from documents with source citations.

## Features
- PDF, TXT, and Markdown document ingestion
- Dense semantic search and BM25 keyword search
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Answer generation using Groq
- Citation verification and confidence scoring
- FastAPI backend and Streamlit dashboard

## Tech Stack
Python, FastAPI, Streamlit, ChromaDB, Sentence Transformers, BM25, Groq

## Setup

**1. Clone the repository**
```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Hybrid_RAG
```

**2. Install dependencies**
```bash
conda create -n hybrid-rag python=3.10
conda activate hybrid-rag
pip install -r requirements.txt
```

**3. Configure the API key**

Create a `.env` file in the project root and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

**4. Run the application**

Start the FastAPI backend:
```bash
uvicorn src.api.main:app --reload
```

In a second terminal, start the Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501** to use the application.

API documentation: **http://127.0.0.1:8000/docs**

## Evaluation
The project includes evaluation for retrieval quality, answer quality, citations, abstention, and response latency.

## Future Improvements
- Incremental document ingestion
- Improved citation verification
- Automated testing

## License
No license has been added yet.