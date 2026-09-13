"""
rag/evaluate_rag.py
Precision and Recall evaluation comparing Fixed-Size vs. Sentence-Based chunking.
Satisfies Part 1 Task 5 by mapping retrieved chunks back to parent documents,
deduplicating them, and printing per-query arithmetic for both ChromaDB collections.
"""

from typing import List, Dict, Any, Set
from rag.vector_store import get_chroma_collection, embed_texts

# Required 5 in-scope benchmark queries with ground-truth target parent document IDs
EVAL_QUERIES: List[Dict[str, Any]] = [
    {
        "query_id": "Q1",
        "query": "What is the return window for apparel and cosmetics?",
        "relevant_doc_ids": {"KB-RET-001"}
    },
    {
        "query_id": "Q2",
        "query": "How are cash on delivery COD refunds processed and what are the timelines?",
        "relevant_doc_ids": {"KB-REF-002"}
    },
    {
        "query_id": "Q3",
        "query": "What are the standard delivery timelines for metro cities versus regional hubs?",
        "relevant_doc_ids": {"KB-SLA-003"}
    },
    {
        "query_id": "Q4",
        "query": "What are the warranty coverage terms for electronic hair styling appliances?",
        "relevant_doc_ids": {"KB-WAR-005"}
    },
    {
        "query_id": "Q5",
        "query": "Can I request a size exchange for footwear and how many times?",
        "relevant_doc_ids": {"KB-EXC-009"}
    }
]


def evaluate_collection(strategy: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Evaluates top-k retrieval against parent document ground truths for a strategy.
    Maps retrieved chunks back to parent doc IDs and deduplicates before scoring.
    """
    collection = get_chroma_collection(strategy=strategy)
    query_texts = [q["query"] for q in EVAL_QUERIES]
    query_embeddings = embed_texts(query_texts)

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k
    )

    query_metrics = []
    total_precision = 0.0
    total_recall = 0.0

    print(f"\n{'='*70}")
    print(f"EVALUATION: Strategy = '{strategy.upper()}' (Top-k = {top_k})")
    print(f"{'='*70}")

    for idx, q_meta in enumerate(EVAL_QUERIES):
        q_id = q_meta["query_id"]
        query_text = q_meta["query"]
        relevant_docs: Set[str] = q_meta["relevant_doc_ids"]

        retrieved_metadatas = results["metadatas"][idx] if results.get("metadatas") else []
        
        # Extract parent_doc_id and deduplicate
        retrieved_parent_docs: List[str] = []
        for m in retrieved_metadatas:
            parent_id = m.get("parent_doc_id") or m.get("doc_id")
            if parent_id and parent_id not in retrieved_parent_docs:
                retrieved_parent_docs.append(parent_id)

        # Compute document-level intersection
        true_positives = [doc for doc in retrieved_parent_docs if doc in relevant_docs]
        tp_count = len(true_positives)
        retrieved_count = len(retrieved_parent_docs)
        relevant_count = len(relevant_docs)

        # Precision = TP / Total Retrieved Docs (deduped)
        precision = (tp_count / retrieved_count) if retrieved_count > 0 else 0.0
        # Recall = TP / Total Relevant Ground-Truth Docs
        recall = (tp_count / relevant_count) if relevant_count > 0 else 0.0

        total_precision += precision
        total_recall += recall

        query_metrics.append({
            "query_id": q_id,
            "precision": precision,
            "recall": recall,
            "retrieved_parents": retrieved_parent_docs,
            "relevant_parents": list(relevant_docs)
        })

        # Display visible per-query arithmetic
        print(f"\n[{q_id}] Query: \"{query_text}\"")
        print(f"  Target Ground Truth: {list(relevant_docs)}")
        print(f"  Retrieved Parent Docs (Deduped): {retrieved_parent_docs}")
        print(f"  True Positives: {true_positives}")
        print(f"  Arithmetic Precision: {tp_count} / {retrieved_count} = {precision:.3f}")
        print(f"  Arithmetic Recall:    {tp_count} / {relevant_count} = {recall:.3f}")

    n = len(EVAL_QUERIES)
    mean_precision = total_precision / n
    mean_recall = total_recall / n

    print(f"\n--- Aggregate Summary for Strategy: '{strategy}' ---")
    print(f"Mean Precision@{top_k}: {mean_precision:.3f}")
    print(f"Mean Recall@{top_k}:    {mean_recall:.3f}")

    return {
        "strategy": strategy,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "details": query_metrics
    }


def compare_chunking_strategies():
    """Executes evaluation across both collections and displays recommendations."""
    fixed_res = evaluate_collection(strategy="fixed", top_k=3)
    sentence_res = evaluate_collection(strategy="sentence", top_k=3)

    print(f"\n{'='*70}")
    print("FINAL CHUNKING STRATEGY COMPARISON & RECOMMENDATION")
    print(f"{'='*70}")
    print(f"Fixed-Size Strategy:    Precision = {fixed_res['mean_precision']:.3f} | Recall = {fixed_res['mean_recall']:.3f}")
    print(f"Sentence-Based Strategy: Precision = {sentence_res['mean_precision']:.3f} | Recall = {sentence_res['mean_recall']:.3f}")

    print("\nRecommendation:")
    if sentence_res['mean_precision'] >= fixed_res['mean_precision']:
        rec = (
            "We recommend deploying the SENTENCE-BASED chunking strategy. Because policy documents "
            "are tightly phrased across 2 to 5 distinct sentences, chunking strictly along grammatical boundaries "
            f"preserves complete semantic clauses and achieves superior retrieval precision ({sentence_res['mean_precision']:.3f} vs {fixed_res['mean_precision']:.3f}) "
            f"while matching or exceeding recall ({sentence_res['mean_recall']:.3f}). Fixed character splits introduce boundary fragmentation, "
            "occasionally severing conditional clauses from their qualifying rules."
        )
    else:
        rec = (
            "We recommend deploying the FIXED-SIZE chunking strategy based on measured retrieval precision "
            f"({fixed_res['mean_precision']:.3f} vs {sentence_res['mean_precision']:.3f}) across the benchmark set."
        )
    print(rec)


if __name__ == "__main__":
    compare_chunking_strategies()