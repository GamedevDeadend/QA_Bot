# 🤖 RAG QA Bot — Chat with your PDF locally

A privacy-first conversational AI chatbot that lets you upload any PDF and have a full conversation about it. Powered by a fully local RAG (Retrieval-Augmented Generation) pipeline — **no API keys, no internet, no data leaves your machine.**

---

## 🎬 Demo

> Upload a PDF → Choose a personality → Ask questions → Get accurate answers instantly

![Demo Screenshot](assets/demo.png)

---

## ✨ Features

- 📄 Upload any PDF document
- 🔍 Intelligent semantic search using vector embeddings
- 🧠 Fully local LLM inference via Ollama — complete privacy
- 💬 Conversational memory — bot remembers context across messages
- 🔄 Smart question condensing — follow-up questions work naturally
- 🌐 DuckDuckGo web search fallback — searches web when PDF has no answer
- 🎭 Three personality modes — Formal, Friendly, Flirtatious
- 🔍 Force web search option — manually trigger web search anytime
- ⚡ Retriever caching — fast responses after first question
- 🖥️ Clean chat UI powered by Gradio
- 🔑 No API keys required

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Ollama](https://ollama.com) | Local LLM inference (Qwen 2.5 3B) |
| [LangChain](https://langchain.com) | RAG pipeline orchestration |
| [ChromaDB](https://trychroma.com) | Vector database for embeddings |
| [nomic-embed-text](https://ollama.com/library/nomic-embed-text) | Local embedding model |
| [PyMuPDF](https://pymupdf.readthedocs.io) | Fast PDF document loading |
| [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search) | Web search fallback |
| [Gradio](https://gradio.app) | Chat web UI |

---

## 🏗️ Architecture

```
PDF Upload
    ↓
PyMuPDFLoader → loads document pages (faster than PyPDF)
    ↓
RecursiveCharacterTextSplitter → chunks text (700 chars, 140 overlap)
    ↓
nomic-embed-text → generates embeddings locally
    ↓
ChromaDB → stores and indexes embeddings
    ↓
Retriever Cache → skips re-processing on follow-up questions
    ↓
User Query → Question Condensing (resolves follow-up context)
    ↓
Semantic Search → top relevant chunks retrieved
    ↓
Personality Prompt → shapes LLM tone and style
    ↓
Qwen 2.5 3B (via Ollama) → generates answer from context
    ↓
Answer Useful? → No → DuckDuckGo Web Search fallback
    ↓
Gradio Chat UI → displays answer with full history
```

---

## 🎭 Personality Modes

| Personality | Description |
|-------------|-------------|
| **Formal Girl** | Professional, precise, authoritative tone |
| **Friendly Girl** | Warm, casual, encouraging — like talking to a friend |
| **Flirtatious Girl** | Playful, witty, charming with light humor |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/download) installed on your machine

### 1. Clone the repository

```bash
git clone https://github.com/GamedevDeadend/QA_Bot.git
cd QA_Bot
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
langchain-core
chromadb
gradio
pymupdf
duckduckgo-search
ddgs
```

---

## 📁 Project Structure

```
QA_Bot/
│
├── Question_Answer_Bot/
│   ├── qabot.py           # Main application
│   └── requirements.txt   # Python dependencies
├── README.md              # You are here
├── LICENSE                # Apache 2.0
└── assets/
    └── demo.png           # Demo screenshot
```

---

## 💡 How It Works

This project uses **RAG (Retrieval-Augmented Generation)** — a technique where instead of relying purely on the LLM's training data, we:

1. Break the PDF into small chunks
2. Convert chunks into vector embeddings using `nomic-embed-text`
3. Store them in ChromaDB vector database
4. When you ask a question, condense it using chat history for context
5. Find the most relevant chunks via semantic search
6. Feed chunks + personality prompt + question to the LLM
7. If answer isn't useful → automatically search DuckDuckGo
8. Return answer with full conversation history

---

## 🔒 Privacy First

Unlike cloud-based solutions (ChatGPT, Claude, Gemini), this app runs **100% on your local machine**:
- No data sent to any server
- No API keys needed
- Works completely offline after setup
- Your documents stay completely private

---

## 👤 Author

**Tanmay Agrawal**
- GitHub: [@GamedevDeadend](https://github.com/GamedevDeadend)
- LinkedIn: [Tanmay Agrawal](https://www.linkedin.com/in/tanmay-agrawal-2954361a0)

---

## 📄 License

[Apache 2.0](LICENSE) — feel free to use and modify.