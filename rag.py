import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

DATA_PATH = "data/books.json"
INDEX_PATH = "data/faiss_index.pkl"
BOOKS_PATH = "data/books_store.pkl"

def load_books():
    with open(DATA_PATH, "r") as f:
        return json.load(f)

def build_document(book):
    """Convert book dict to a rich text document for embedding."""
    genres = ", ".join(book["genre"])
    moods = ", ".join(book["mood"])
    themes = ", ".join(book["themes"])
    return (
        f"Title: {book['title']} by {book['author']}. "
        f"Genre: {genres}. "
        f"Mood: {moods}. "
        f"Themes: {themes}. "
        f"Description: {book['description']} "
        f"Published: {book['year']}. Rating: {book['rating']}/5. Pages: {book['pages']}."
    )

def build_index():
    """Build FAISS index from books data."""
    books = load_books()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    documents = [build_document(book) for book in books]
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity after normalization
    index.add(embeddings)
    
    # Save index and books
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    with open(BOOKS_PATH, "wb") as f:
        pickle.dump({"books": books, "documents": documents}, f)
    
    print(f"Index built with {len(books)} books.")
    return index, books, documents

def load_index():
    """Load existing FAISS index or build if not exists."""
    if os.path.exists(INDEX_PATH) and os.path.exists(BOOKS_PATH):
        with open(INDEX_PATH, "rb") as f:
            index = pickle.load(f)
        with open(BOOKS_PATH, "rb") as f:
            store = pickle.load(f)
        return index, store["books"], store["documents"]
    else:
        return build_index()

def retrieve(query, index, books, documents, model, top_k=3):
    """Retrieve top-k most relevant books for a query."""
    query_embedding = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_embedding)
    
    scores, indices = index.search(query_embedding, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        book = books[idx]
        results.append({
            "book": book,
            "document": documents[idx],
            "score": float(score)
        })
    return results

if __name__ == "__main__":
    build_index()
    print("Done!")
