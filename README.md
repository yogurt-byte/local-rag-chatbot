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
#**Getting Started**
**Prerequisites**
Install Python 3.9+
Download and install Ollama from ollama.com
Download the local models by running the following commands in your terminal:
bash


ollama pull nomic-embed-text
ollama pull llama3.2
**Installation & Setup**
Clone the repository:

bash


git clone https://github.com/yogurt-byte/local-rag-chatbot.git
cd local-rag-chatbot
Activate the virtual environment:

PowerShell:
powershell


.\venv\Scripts\Activate.ps1
Command Prompt:
cmd


.\venv\Scripts\activate.bat
Install the dependencies:

bash


pip install langchain langchain-community langchain-chroma langchain-ollama python-dotenv
Add your documents: Create a folder named docs (if not already present) and add the text documents (.txt files) you want your AI to know about.

**How to Run**
Step 1: Ingest Documents
Process the raw text files, generate embeddings, and save them to the database:

bash


python 1_ingestion_pipeline.py
Step 2: Test Retrieval
Run a test query to verify that document retrieval and basic model completion are working properly:

bash


python 2_retrieval_pipeline.py
Step 3: Run the Chatbot
Start the interactive chatbot terminal session:

bash


python 3_history_aware_generation.py
Ask questions based on your documents, and type quit to exit.

....then what is this for


