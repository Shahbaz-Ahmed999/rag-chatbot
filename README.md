# 🏦 Pakistani Finance Q&A Assistant (RAG Chatbot)

A domain-specific Retrieval Augmented Generation (RAG) chatbot that answers questions about Pakistan's economy, banking, and capital markets using real documents — not guesswork.

Built with LangChain, ChromaDB, Groq LLaMA-3, and Streamlit.

---

## 🖥️ Live Demo
[Click here to try the live app](https://shahbaz-ahmed999-rag-chatbot-app-evf41x.streamlit.app/)

---

## 📌 What This Project Does

- Ingests Pakistani financial documents (PDFs)
- Splits them into 482 searchable chunks
- Converts chunks into semantic vectors using HuggingFace embeddings
- Stores vectors in a local ChromaDB vector database
- Accepts user questions and retrieves the most relevant chunks
- Sends retrieved context + question to LLaMA-3 via Groq API
- Returns accurate, document-grounded answers with source citation

---

## 🧠 How RAG Works
PDF Documents → Text Chunks → Embeddings → ChromaDB
↓
User Question → Embedding → Similarity Search → Top 4 Chunks
↓
LLaMA-3 (Groq) → Grounded Answer + Source

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| LangChain | Pipeline orchestration |
| ChromaDB | Local vector database |
| Groq LLaMA-3 | Free LLM API |
| HuggingFace Embeddings | Text to vector conversion |
| Streamlit | Web interface |
| PyPDF | PDF document loading |

---

## 📂 Project Structure

    rag-chatbot/
    ├── app.py                  ← Streamlit web interface
    ├── documents/              ← PDF source documents
    ├── src/
    │   ├── loader.py           ← PDF loading and chunking
    │   ├── vectorstore.py      ← ChromaDB vector store
    │   └── rag_chain.py        ← LLM chain and retrieval
    ├── vectorstore_db/         ← Saved ChromaDB vectors
    ├── .env                    ← API keys (not pushed to GitHub)
    └── requirements.txt        ← Python dependencies

---

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Create a `.env` file in the root folder:

GROQ_API_KEY=your_groq_api_key_here
Get a free key at: https://console.groq.com

**5. Add your PDF documents**

Place PDF files inside the `documents/` folder, then build the vector store:
```bash
python src/vectorstore.py
```

**6. Run the app**
```bash
streamlit run app.py
```

---

## 💡 Key Learnings

- RAG grounds LLM responses in real documents, eliminating hallucination
- Chunk size (500) and overlap (50) significantly affect retrieval quality
- Document quality matters — scanned PDFs produce poor embeddings
- "I don't know" is the correct answer when context is missing

---

## 👤 Author

Shahbaz Ahmed Khan
[LinkedIn](#) | [GitHub](#)

