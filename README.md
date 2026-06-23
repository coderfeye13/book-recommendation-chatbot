# 📚 Book Recommendation Chatbot

A RAG-powered chatbot that recommends books based on your mood, interests, or preferences.

Built for the **OpenCampus "From LLMs to AI Agents"** course — Summer 2026.

## 🏗️ Architecture

```
User Query → Embedding (sentence-transformers) → FAISS Vector Search → Retrieved Books
                                                                              ↓
                                                          LLM (LLaMA 3.1 via Groq) → Response
```

**RAG Pipeline:**
1. Book descriptions are embedded using `sentence-transformers/all-MiniLM-L6-v2`
2. Embeddings stored in a FAISS index with cosine similarity search
3. User query is embedded and top-3 most relevant books are retrieved
4. LLaMA 3.1 generates a personalized recommendation using the retrieved context
5. Conversation history is maintained for follow-up questions

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | LLaMA 3.1 8B (via Groq API) |
| Embeddings | `all-MiniLM-L6-v2` (local, free) |
| Vector Store | FAISS (cosine similarity) |
| Framework | LangChain concepts, custom RAG |
| UI | Streamlit |
| Deploy | Streamlit Cloud |

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/book-recommendation-chatbot
cd book-recommendation-chatbot
pip install -r requirements.txt

# Add your Groq API key
mkdir -p .streamlit
echo 'GROQ_API_KEY = "your_key_here"' > .streamlit/secrets.toml

streamlit run app.py
```

## 💬 Example Queries

- *"I want something funny and lighthearted"*
- *"Recommend a dark dystopian novel"*
- *"I'm a software engineer, what should I read?"*
- *"Something inspiring for when I'm feeling lost"*
- *"I loved Dune, what's similar?"*
