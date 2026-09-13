"""
rag/vector_store.py
Manages ChromaDB vector indexing and local SentenceTransformer embeddings.
Satisfies Part 1 Task 3:
- Embeds chunks using a local, free SentenceTransformers model.
- Indexes each chunking strategy into its own separate ChromaDB collection via collection.upsert().
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from data.knowledge_base import load_documents_for_indexing
from rag.chunking import prepare_chunks

# Constants and local persistence paths
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

COLLECTION_FIXED = "nykaa_kb_fixed"
COLLECTION_SENTENCE = "nykaa_kb_sentence"

# Global local embedding model singleton
_EMBED_MODEL: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Loads and caches the local SentenceTransformer model."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBED_MODEL


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Encodes a list of strings into dense vector representations."""
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


def get_chroma_client() -> chromadb.ClientAPI:
    """Initializes and returns a persistent ChromaDB client."""
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )


def get_chroma_collection(strategy: str = "sentence"):
    """
    Fetches the ChromaDB collection corresponding to the chunking strategy.
    
    Args:
        strategy: 'fixed' or 'sentence'
    """
    client = get_chroma_client()
    collection_name = COLLECTION_FIXED if strategy == "fixed" else COLLECTION_SENTENCE
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "strategy": strategy}
    )


def index_knowledge_base(strategy: str) -> int:
    """
    Loads raw knowledge base documents, chunks them by the chosen strategy,
    computes embeddings, and upserts them into their separate ChromaDB collection.
    
    Args:
        strategy: 'fixed' or 'sentence'
    Returns:
        Number of chunks indexed.
    """
    raw_docs = load_documents_for_indexing()
    chunks = prepare_chunks(raw_docs, strategy=strategy)

    if not chunks:
        return 0

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    
    # Generate local embeddings
    embeddings = embed_texts(texts)

    collection = get_chroma_collection(strategy=strategy)
    
    # Upsert chunks into the strategy's dedicated collection
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    return len(chunks)


def build_both_collections() -> Dict[str, int]:
    """Indexes both chunking collections sequentially."""
    print("Indexing fixed-size chunking collection...")
    fixed_count = index_knowledge_base(strategy="fixed")

    print("Indexing sentence-based chunking collection...")
    sentence_count = index_knowledge_base(strategy="sentence")

    return {
        "fixed_chunks_indexed": fixed_count,
        "sentence_chunks_indexed": sentence_count
    }


if __name__ == "__main__":
    print("--- Building Dual ChromaDB Collections (Part 1 Task 3) ---")
    counts = build_both_collections()
    print(f"Indexing Complete!")
    print(f"Collection '{COLLECTION_FIXED}': {counts['fixed_chunks_indexed']} chunks.")
    print(f"Collection '{COLLECTION_SENTENCE}': {counts['sentence_chunks_indexed']} chunks.")

    # Quick sanity query on the sentence collection
    sample_query = "What is the return window for cosmetics?"
    sent_collection = get_chroma_collection(strategy="sentence")
    query_vec = embed_texts([sample_query])
    
    sample_res = sent_collection.query(query_embeddings=query_vec, n_results=1)
    print(f"\n[Sanity Check Query]: '{sample_query}'")
    print(f"Top Retrieved Chunk: {sample_res['documents'][0][0]}")