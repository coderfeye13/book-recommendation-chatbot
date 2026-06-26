import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from rag import load_index, retrieve

# ── Page config 
st.set_page_config(
    page_title="📚 Book Recommendation Chatbot",
    page_icon="📚",
    layout="centered"
)

# ── Custom CSS
st.markdown("""
<style>
    .book-card {
        background: #1e1e2e;
        border-left: 4px solid #cba6f7;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .book-title { color: #cba6f7; font-weight: bold; font-size: 1rem; }
    .book-meta { color: #a6adc8; font-size: 0.8rem; margin-top: 4px; }
    .source-label { color: #6c7086; font-size: 0.75rem; margin-top: 12px; }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Load models (cached) 
@st.cache_resource(show_spinner="Loading book database...")
def load_resources():
    index, books, documents = load_index()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, books, documents, embedding_model

index, books, documents, embedding_model = load_resources()

# ── Groq client 
groq_api_key = st.secrets.get("GROQ_API_KEY", None)
if not groq_api_key:
    import os
    groq_api_key = os.environ.get("GROQ_API_KEY", "")

client = Groq(api_key=groq_api_key)
MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are an enthusiastic and knowledgeable book recommendation assistant. 
Your job is to help users discover books they'll love based on their interests, mood, or preferences.

You have access to a curated book database. Based on the retrieved books provided to you as context, 
give personalized, conversational recommendations. 

Guidelines:
- Be warm, friendly, and enthusiastic about books
- Explain WHY a book matches what the user is looking for
- Mention specific themes, mood, or elements that align with the user's request
- Keep responses concise but informative (2-4 sentences per book)
- If the user asks follow-up questions, remember the conversation history
- If no books match well, be honest and ask for more details
- Always base recommendations on the provided context books"""

def format_context(retrieved_books):
    """Format retrieved books as context for the LLM."""
    context = "Here are the most relevant books from the database:\n\n"
    for i, result in enumerate(retrieved_books, 1):
        b = result["book"]
        context += f"{i}. **{b['title']}** by {b['author']}\n"
        context += f"   Genre: {', '.join(b['genre'])}\n"
        context += f"   Mood: {', '.join(b['mood'])}\n"
        context += f"   Rating: {b['rating']}/5 | Pages: {b['pages']} | Year: {b['year']}\n"
        context += f"   {b['description']}\n\n"
    return context

SIMILARITY_THRESHOLD = 0.30  # minimum relevance score

def get_response(user_message, chat_history):
    """Get LLM response with RAG context."""
    # Retrieve relevant books
    retrieved = retrieve(user_message, index, books, documents, embedding_model, top_k=3)
    
    # Check if any book meets the threshold
    good_matches = [r for r in retrieved if r["score"] >= SIMILARITY_THRESHOLD]
    
    if not good_matches:
        best_score = int(retrieved[0]["score"] * 100)
        fallback = (
            f"I couldn't find a strong match for your request in my book database "
            f"(best relevance score: {best_score}%). \n\n"
            "Try describing a **mood**, **genre**, or **theme** instead — for example:\n"
            "- *'Something dark and dystopian'*\n"
            "- *'An inspiring non-fiction book'*\n"
            "- *'I loved Dune, what's similar?'*"
        )
        return fallback, retrieved
    
    context = format_context(good_matches)
    
    # Build messages for API
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({
        "role": "user",
        "content": f"{context}\n\nUser question: {user_message}"
    })
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=800,
        temperature=0.7,
    )
    
    return response.choices[0].message.content, retrieved


    """Get LLM response with RAG context."""
    # Retrieve relevant books
    retrieved = retrieve(user_message, index, books, documents, embedding_model, top_k=3)
    context = format_context(retrieved)
    
    # Build messages for API
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add chat history (last 6 messages for context window)
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current message with RAG context
    messages.append({
        "role": "user",
        "content": f"{context}\n\nUser question: {user_message}"
    })
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=800,
        temperature=0.7,
    )
    
    return response.choices[0].message.content, retrieved

# ── UI 
st.title("📚 Book Recommendation Chatbot")
st.caption("Tell me what you're in the mood for, and I'll find the perfect book for you!")

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.write("""
    This chatbot uses **RAG (Retrieval-Augmented Generation)** to recommend books.
    
    **How it works:**
    1. Your query is converted to an embedding
    2. Similar books are retrieved from the vector database (FAISS)
    3. An LLM generates a personalized recommendation based on the retrieved books
    """)
    
    st.divider()
    st.write("**Tech Stack:**")
    st.write("- 🧠 LLM: LLaMA 3.1 (via Groq)")
    st.write("- 🔍 Embeddings: sentence-transformers")
    st.write("- 📦 Vector Store: FAISS")
    st.write("- 🎨 UI: Streamlit")
    
    st.divider()
    st.write("**Try asking:**")
    st.write("- *I want something funny and light*")
    st.write("- *Recommend a dark dystopian novel*")
    st.write("- *I'm a software engineer, what should I read?*")
    st.write("- *Something inspiring for when I'm feeling lost*")
    st.write("- *I loved Dune, what's similar?*")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieved_books" not in st.session_state:
    st.session_state.retrieved_books = {}

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show retrieved books as sources for assistant messages
        if message["role"] == "assistant" and i in st.session_state.retrieved_books:
            with st.expander("📖 Books retrieved from database", expanded=False):
                for result in st.session_state.retrieved_books[i]:
                    b = result["book"]
                    score_pct = int(result["score"] * 100)
                    st.markdown(f"""
                    <div class="book-card">
                        <div class="book-title">📗 {b['title']} — {b['author']}</div>
                        <div class="book-meta">
                            {' | '.join(b['genre'])} &nbsp;·&nbsp; 
                            ⭐ {b['rating']} &nbsp;·&nbsp; 
                            {b['pages']} pages &nbsp;·&nbsp; 
                            Relevance: {score_pct}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("What kind of book are you looking for?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Finding the perfect books for you..."):
            response_text, retrieved = get_response(prompt, st.session_state.messages[:-1])
        
        st.markdown(response_text)
        
        # Store retrieved books for this message
        msg_idx = len(st.session_state.messages)
        st.session_state.retrieved_books[msg_idx] = retrieved
        
        with st.expander("📖 Books retrieved from database", expanded=False):
            for result in retrieved:
                b = result["book"]
                score_pct = int(result["score"] * 100)
                st.markdown(f"""
                <div class="book-card">
                    <div class="book-title">📗 {b['title']} — {b['author']}</div>
                    <div class="book-meta">
                        {' | '.join(b['genre'])} &nbsp;·&nbsp; 
                        ⭐ {b['rating']} &nbsp;·&nbsp; 
                        {b['pages']} pages &nbsp;·&nbsp; 
                        Relevance: {score_pct}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# Welcome message if no chat yet
if not st.session_state.messages:
    st.info("👋 Hi! I'm your personal book recommendation assistant. Tell me what you're in the mood for — a genre, a feeling, a theme, or even a book you loved — and I'll suggest something perfect for you!")
