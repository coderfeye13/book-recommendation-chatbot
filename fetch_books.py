"""
Fetch books from Open Library API and convert to our books.json format.
Run this locally: python fetch_books.py
"""

import requests
import json
import time

SEARCH_URL = "https://openlibrary.org/search.json"
FIELDS = "title,author_name,subject,first_publish_year,ratings_average,number_of_pages_median,isbn"

# Search queries to cover different genres
QUERIES = [
    ("thriller mystery", "Thriller", ["suspenseful", "tense", "gripping"]),
    ("romance love story", "Romance", ["romantic", "emotional", "heartwarming"]),
    ("horror stephen king", "Horror", ["dark", "scary", "atmospheric"]),
    ("biography memoir inspiring", "Biography", ["inspiring", "emotional", "eye-opening"]),
    ("fantasy magic adventure", "Fantasy", ["immersive", "magical", "adventurous"]),
    ("artificial intelligence technology future", "Technology", ["educational", "thought-provoking", "futuristic"]),
    ("philosophy stoicism life", "Philosophy", ["reflective", "profound", "enlightening"]),
    ("psychology behavior human mind", "Psychology", ["enlightening", "analytical", "thought-provoking"]),
]

# Genre → mood mapping
GENRE_MOOD_MAP = {
    "Thriller": ["suspenseful", "tense", "gripping", "intense"],
    "Romance": ["romantic", "emotional", "heartwarming", "hopeful"],
    "Horror": ["dark", "scary", "atmospheric", "tense"],
    "Biography": ["inspiring", "emotional", "eye-opening", "reflective"],
    "Fantasy": ["immersive", "magical", "adventurous", "epic"],
    "Technology": ["educational", "thought-provoking", "futuristic", "practical"],
    "Philosophy": ["reflective", "profound", "enlightening", "thought-provoking"],
    "Psychology": ["enlightening", "analytical", "thought-provoking", "practical"],
}

def extract_themes(subjects):
    """Extract meaningful themes from Open Library subjects."""
    if not subjects:
        return ["general", "literature"]
    
    theme_keywords = [
        "love", "war", "death", "survival", "identity", "family", "friendship",
        "power", "freedom", "justice", "science", "technology", "nature", "history",
        "religion", "society", "psychology", "philosophy", "adventure", "mystery",
        "politics", "culture", "art", "music", "education", "economics", "ethics",
        "leadership", "creativity", "innovation", "trauma", "resilience", "memory",
        "time", "space", "human nature", "morality", "consciousness", "relationships"
    ]
    
    found = []
    subjects_lower = [s.lower() for s in subjects[:20]]
    
    for keyword in theme_keywords:
        if any(keyword in s for s in subjects_lower):
            found.append(keyword)
        if len(found) >= 5:
            break
    
    return found if found else ["general fiction", "literature"]

def extract_genre(subjects, default_genre):
    """Extract genre from subjects."""
    genre_map = {
        "Fiction": "Fiction",
        "Science fiction": "Science Fiction",
        "Fantasy fiction": "Fantasy",
        "Mystery": "Mystery",
        "Thriller": "Thriller",
        "Horror": "Horror",
        "Romance": "Romance",
        "Biography": "Biography",
        "History": "History",
        "Philosophy": "Philosophy",
        "Psychology": "Psychology",
        "Self-help": "Self-Help",
        "Business": "Business",
        "Technology": "Technology",
        "Non-fiction": "Non-Fiction",
    }
    
    genres = [default_genre]
    if subjects:
        for subj in subjects[:15]:
            for key, val in genre_map.items():
                if key.lower() in subj.lower() and val not in genres:
                    genres.append(val)
                    break
    
    return genres[:3]

def fetch_books_for_query(query, default_genre, default_mood, limit=5):
    """Fetch books from Open Library for a given query."""
    params = {
        "q": query,
        "limit": limit * 3,  # fetch more, filter down
        "fields": FIELDS,
        "language": "eng",
    }
    
    try:
        resp = requests.get(SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Error fetching '{query}': {e}")
        return []
    
    books = []
    for doc in data.get("docs", []):
        title = doc.get("title", "").strip()
        authors = doc.get("author_name", [])
        year = doc.get("first_publish_year")
        rating = doc.get("ratings_average")
        pages = doc.get("number_of_pages_median")
        subjects = doc.get("subject", [])
        
        # Filter: must have title, author, year, rating
        if not title or not authors or not year or not rating:
            continue
        if rating < 3.5:
            continue
        if pages and pages < 50:
            continue
        
        author = ", ".join(authors[:2])
        genre = extract_genre(subjects, default_genre)
        themes = extract_themes(subjects)
        mood = GENRE_MOOD_MAP.get(default_genre, default_mood)
        
        book = {
            "title": title,
            "author": author,
            "genre": genre,
            "mood": mood,
            "description": f"A highly rated {genre[0].lower()} work by {author}. "
                          f"Published in {year}, this book explores themes of {', '.join(themes[:3])}.",
            "themes": themes,
            "pages": pages if pages else 300,
            "year": year,
            "rating": round(rating, 1),
        }
        books.append(book)
        
        if len(books) >= limit:
            break
    
    return books

def main():
    print("Fetching books from Open Library API...")
    print("=" * 50)
    
    all_books = []
    seen_titles = set()
    
    for query, genre, mood in QUERIES:
        print(f"\nFetching: {query}")
        books = fetch_books_for_query(query, genre, mood, limit=5)
        
        for book in books:
            if book["title"].lower() not in seen_titles:
                seen_titles.add(book["title"].lower())
                all_books.append(book)
                print(f"  ✓ {book['title']} — {book['author']} ({book['year']}) ⭐{book['rating']}")
        
        time.sleep(0.5)  # be nice to the API
    
    print(f"\n{'='*50}")
    print(f"Total new books fetched: {len(all_books)}")
    
    # Load existing books.json
    try:
        with open("data/books.json", "r") as f:
            existing = json.load(f)
        print(f"Existing books: {len(existing)}")
        
        # Filter out duplicates with existing
        existing_titles = {b["title"].lower() for b in existing}
        new_books = [b for b in all_books if b["title"].lower() not in existing_titles]
        print(f"New unique books to add: {len(new_books)}")
        
        combined = existing + new_books
    except FileNotFoundError:
        print("No existing books.json found, creating new one.")
        combined = all_books
    
    # Save
    with open("data/books.json", "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ books.json updated: {len(combined)} total books")
    print("\nDon't forget to delete data/faiss_index.pkl and data/books_store.pkl")
    print("so the index gets rebuilt on next app start!")

if __name__ == "__main__":
    main()
