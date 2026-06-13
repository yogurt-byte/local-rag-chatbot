# Local RAG Chatbot with History-Aware Retrieval

A privacy-first, fully local Retrieval-Augmented Generation (RAG) chatbot built using **Python**, **LangChain**, **Chroma DB**, and **Ollama**. This project allows you to ingest a directory of text documents, index them into a persistent vector database, and hold multi-turn conversations with a local LLM that understands the context of your files.

---

## Key Features

* **100% Offline & Private:** Zero API calls or external network requests. All embeddings, storage, and text generation run entirely on your local machine using Ollama.
* **Semantic Vector Search:** Automatically loads, chunks, and indexes custom text documents (`docs/`) into a persistent **Chroma DB** using **nomic-embed-text** embeddings.
* **Diverse Context Retrieval:** Utilizes **Maximum Marginal Relevance (MMR)** search query logic to fetch highly relevant context chunks while reducing redundant information.
* **Conversational Memory:** Designed a custom history-aware query reformulation pipeline that converts multi-turn chat interactions into standalone search queries for accurate database lookups.

---

## Tech Stack

* **Language:** Python
* **Orchestration:** LangChain (langchain-community, langchain-ollama, langchain-chroma)
* **Vector Database:** Chroma DB
* **Local Models (Ollama):** 
  * LLM: `llama3.2:3b`
  * Embeddings: `nomic-embed-text`
* **Configuration:** python-dotenv

---

## Project Structure

```text
├── docs/                           # Put your custom .txt files here
├── db/                             # Auto-generated persistent Chroma vector store
├── 1_ingestion_pipeline.py         # Loads documents, chunks text, creates embeddings
├── 2_retrieval_pipeline.py         # CLI testing for single query search & retrieval
├── 3_history_aware_generation.py   # Multi-turn chatbot with memory and query rewriting
├── .gitignore                      # Prevents committing venv, DB, and API keys
└── README.md                       # Project documentation
```



