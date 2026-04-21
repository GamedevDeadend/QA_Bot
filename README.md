# 🤖 RAG QA Bot — Chat with your PDF locally

A privacy-first Question Answering chatbot that lets you upload any PDF and ask questions about it. Powered by a fully local RAG (Retrieval-Augmented Generation) pipeline — **no API keys, no internet, no data leaves your machine.**

---

## 🎬 Demo

> Upload a PDF → Ask a question → Get accurate answers instantly

![Demo Screenshot](assets/demo.png)

---

## ✨ Features

- 📄 Upload any PDF document
- 🔍 Intelligent semantic search using vector embeddings
- 🧠 Fully local LLM inference via Ollama — complete privacy
- ⚡ Fast retrieval using ChromaDB vector store
- 🖥️ Clean web UI powered by Gradio
- 🔑 No API keys required

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Ollama](https://ollama.com) | Local LLM inference (Qwen 2.5 3B) |
| [LangChain](https://langchain.com) | RAG pipeline orchestration |
| [ChromaDB](https://trychroma.com) | Vector database for embeddings |
| [nomic-embed-text](https://ollama.com/library/nomic-embed-text) | Local embedding model |
| [Gradio](https://gradio.app) | Web UI |
| [PyPDFLoader](https://python.langchain.com) | PDF document loading |

---

## 🏗️ Architecture

```
PDF Upload
    ↓
PyPDFLoader → loads document pages
    ↓
RecursiveCharacterTextSplitter → chunks text (1000 chars, 200 overlap)
    ↓
nomic-embed-text → generates embeddings locally
    ↓
ChromaDB → stores and indexes embeddings
    ↓
User Query → semantic search → top relevant chunks retrieved
    ↓
Qwen 2.5 3B (via Ollama) → generates answer from context
    ↓
Gradio UI → displays answer
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/download) installed on your machine

### 1. Clone the repository

```bash
git clone https://github.com/GamedevDeadend/pdf-qa-bot.git
cd pdf-qa-bot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull required Ollama models

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### 4. Run the app

```bash
python qabot.py
```

Open your browser at `http://127.0.0.1:7860`

---

## 📦 Requirements

```
langchain
langchain-ollama
langchain-community
langchain-text-splitters
chromadb
gradio
pypdf
```

---

## 📁 Project Structure

```
QA_Bot/
│
├── qabot.py              # Main application
├── requirements.txt    # Python dependencies
├── README.md           # You are here
└── assets/
    └── demo.png        # Demo screenshot
```

---

## 💡 How It Works

This project uses **RAG (Retrieval-Augmented Generation)** — a technique where instead of relying purely on the LLM's training data, we:

1. Break the PDF into small chunks
2. Convert chunks into vector embeddings
3. Store them in ChromaDB
4. When you ask a question, find the most relevant chunks
5. Feed those chunks + your question to the LLM
6. LLM answers based on your actual document

This means the bot answers from **your document**, not from general knowledge.

---

## 🔒 Privacy First

Unlike cloud-based solutions (ChatGPT, Claude, Gemini), this app runs **100% on your local machine**:
- No data sent to any server
- No API keys needed
- Works completely offline after setup
- Your documents stay private

---

## 👤 Author

**Tanmay Agrawal**
- GitHub: [@GamedevDeadend](https://github.com/GamedevDeadend)
- LinkedIn: [Tanmay Agrawal](https://www.linkedin.com/in/tanmay-agrawal-2954361a0)

---

## 📄 License

Apache2.0 — feel free to use and modify.
