# 🔍 Source-Code-Analysis-RAG

An intelligent Retrieval-Augmented Generation (RAG) system designed to explore, explain, and analyze large codebases. This tool allows developers to "chat" with their source code to find bugs, understand architectural patterns, and generate documentation.

---

## 🛠️ How It Works

This system converts a local or remote repository into a searchable vector database, allowing an LLM (via Groq or OpenAI) to retrieve relevant code snippets and provide context-aware answers.

1. **Ingestion:** Recursively crawls the repository for source files (`.py`, `.js`, `.cpp`, etc.).
2. **Chunking:** Splits code into logical blocks while maintaining function/class context.
3. **Embedding:** Generates vector embeddings using models like `text-embedding-3-small` or HuggingFace.
4. **Vector Store:** Stores code vectors in **ChromaDB** or **Pinecone**.
5. **RAG Pipeline:** Retrieves relevant code blocks based on user queries and processes them through **Llama 3.3 70B** for analysis.

---

## 🚀 Key Features

* **Deep Semantic Search:** Find where specific logic is implemented without knowing the exact variable names.
* **Architecture Mapping:** Ask questions like "How does the authentication flow work in this project?"
* **Refactoring Suggestions:** Identify code smells and get AI-powered suggestions for optimization.
* **Multi-Language Support:** Handles Python, JavaScript, TypeScript, C++, and Java.

---

## ⚙️ Quick Start

### 1. Installation

```bash
git clone [https://github.com/Ahmed2797/-Source-Code-Analysis-RAG]
cd Source-Code-Analysis-RAG
pip install -r requirements.txt
python store_pincone.py
python app.py
