"""
rag/chunking.py
Implements the dual chunking strategies required by Part 1 Task 3:
1. Fixed-size chunking with configurable overlap.
2. Sentence-based boundary chunking.
Maps every generated chunk back to its parent document ID for downstream deduplication.
"""

import re
from typing import List, Dict, Any


def chunk_fixed_size(text: str, chunk_size: int = 180, overlap: int = 35) -> List[str]:
    """
    Chunks input text using a sliding character window with overlap.
    
    Args:
        text: Source policy text.
        chunk_size: Window length in characters.
        overlap: Step back in characters to preserve context across boundaries.
        
    Returns:
        List of non-empty text chunks.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Advance pointer by step size (chunk_size - overlap)
        if end >= text_len:
            break
        start += max(1, chunk_size - overlap)

    return chunks


def chunk_sentence_based(text: str) -> List[str]:
    """
    Splits policy text cleanly along natural sentence terminators (. ! ?).
    Preserves whole semantic statements (ideal for 2-5 sentence policy docs).
    
    Args:
        text: Source policy text.
        
    Returns:
        List of trimmed individual sentences.
    """
    if not text:
        return []

    # Regex splits on sentence-ending punctuation followed by whitespace
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw_sentences if s.strip()]


def prepare_chunks(
    documents: List[Dict[str, Any]], 
    strategy: str = "sentence",
    chunk_size: int = 180,
    overlap: int = 35
) -> List[Dict[str, Any]]:
    """
    Chunks a list of source documents and packages each chunk with unique IDs
    and parent metadata for ChromaDB indexing.
    
    Args:
        documents: List of dicts with keys 'id', 'text', and 'metadata'.
        strategy: 'fixed' or 'sentence'.
        chunk_size: Character length if strategy is 'fixed'.
        overlap: Character overlap if strategy is 'fixed'.
        
    Returns:
        List of structured chunk records with parent_doc_id.
    """
    if strategy not in {"fixed", "sentence"}:
        raise ValueError(f"Unknown chunking strategy: '{strategy}'. Choose 'fixed' or 'sentence'.")

    chunked_records: List[Dict[str, Any]] = []

    for doc in documents:
        doc_id = doc["id"]
        doc_text = doc["text"]
        parent_meta = doc.get("metadata", {})

        if strategy == "fixed":
            raw_chunks = chunk_fixed_size(doc_text, chunk_size=chunk_size, overlap=overlap)
        else:
            raw_chunks = chunk_sentence_based(doc_text)

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{doc_id}_{strategy}_{idx}"
            chunked_records.append({
                "chunk_id": chunk_id,
                "parent_doc_id": doc_id,
                "text": chunk_text,
                "metadata": {
                    **parent_meta,
                    "parent_doc_id": doc_id,
                    "strategy": strategy,
                    "chunk_index": idx
                }
            })

    return chunked_records


if __name__ == "__main__":
    from data.knowledge_base import load_documents_for_indexing

    raw_docs = load_documents_for_indexing()
    print(f"Loaded {len(raw_docs)} parent documents.")

    # Generate chunks under both strategies
    fixed_chunks = prepare_chunks(raw_docs, strategy="fixed")
    sentence_chunks = prepare_chunks(raw_docs, strategy="sentence")

    print(f"\n--- Fixed-Size Chunking (size=180, overlap=35) ---")
    print(f"Total Chunks: {len(fixed_chunks)}")
    print(f"Sample Chunk 0 ID: {fixed_chunks[0]['chunk_id']}")
    print(f"Sample Chunk 0 Parent: {fixed_chunks[0]['parent_doc_id']}")
    print(f"Sample Text: '{fixed_chunks[0]['text']}'")

    print(f"\n--- Sentence-Based Chunking ---")
    print(f"Total Chunks: {len(sentence_chunks)}")
    print(f"Sample Chunk 0 ID: {sentence_chunks[0]['chunk_id']}")
    print(f"Sample Chunk 0 Parent: {sentence_chunks[0]['parent_doc_id']}")
    print(f"Sample Text: '{sentence_chunks[0]['text']}'")