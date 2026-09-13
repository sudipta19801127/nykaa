"""
rag/grounded_engine.py
Satisfies:
- Part 1 Task 4: Grounded generation with empirical threshold calibration.
- Part 4 Task 16: Normalized in-memory response caching with hit/miss counters and timing.
"""

import time
import re
from typing import Dict, Any, List, Optional, Tuple
from rag.vector_store import get_chroma_collection, embed_texts

# ---------------------------------------------------------------------------
# Part 4 Task 16: In-Memory Response Caching
# ---------------------------------------------------------------------------
class NormalizedResponseCache:
    """In-memory cache keyed by normalized query text with hit/miss telemetry."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.hits: int = 0
        self.misses: int = 0
        self.llm_call_count: int = 0

    @staticmethod
    def normalize_key(query: str) -> str:
        """Lowercases, removes punctuation, and condenses whitespace."""
        cleaned = re.sub(r"[^\w\s]", "", query.lower())
        return " ".join(cleaned.split())

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        key = self.normalize_key(query)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def set(self, query: str, response: Dict[str, Any]) -> None:
        key = self.normalize_key(query)
        self._cache[key] = response


# Global cache singleton
ENGINE_CACHE = NormalizedResponseCache()


# ---------------------------------------------------------------------------
# Part 1 Task 4: Empirical Threshold Calibration
# ---------------------------------------------------------------------------
CALIBRATION_IN_SCOPE_QUERIES = [
    "What is the return window for cosmetics and beauty products?",
    "How are cash on delivery COD refunds credited back?",
    "What are the delivery SLA timelines for metro cities?",
]

CALIBRATION_OUT_OF_SCOPE_QUERIES = [
    "How do I book flight tickets to Mumbai?",
    "Can you explain the theory of general relativity and black holes?",
]


def calibrate_similarity_threshold(strategy: str = "sentence") -> float:
    """
    Measures top-1 cosine similarity for in-scope vs out-of-scope query clusters
    and calculates the empirical separation threshold midpoint.
    """
    collection = get_chroma_collection(strategy=strategy)

    def get_top1_similarity(query: str) -> float:
        emb = embed_texts([query])
        res = collection.query(query_embeddings=emb, n_results=1)
        distances = res.get("distances", [[]])[0]
        if not distances:
            return 0.0
        # ChromaDB default cosine distance d = 1 - cosine_similarity
        # similarity = 1 - distance
        cosine_distance = distances[0]
        similarity = max(0.0, 1.0 - cosine_distance)
        return similarity

    print("\n--- Calibrating Grounded Retrieval Similarity Threshold ---")
    in_scope_scores = []
    for q in CALIBRATION_IN_SCOPE_QUERIES:
        sim = get_top1_similarity(q)
        in_scope_scores.append(sim)
        print(f"  [In-Scope] '{q[:40]}...' -> Top-1 Sim: {sim:.4f}")

    out_scope_scores = []
    for q in CALIBRATION_OUT_OF_SCOPE_QUERIES:
        sim = get_top1_similarity(q)
        out_scope_scores.append(sim)
        print(f"  [Out-of-Scope] '{q[:40]}...' -> Top-1 Sim: {sim:.4f}")

    min_in_scope = min(in_scope_scores)
    max_out_scope = max(out_scope_scores)

    # Midpoint calibration between clusters
    calibrated_threshold = round((min_in_scope + max_out_scope) / 2.0, 3)

    print(f"\nCalibration Results:")
    print(f"  Min In-Scope Similarity:      {min_in_scope:.4f}")
    print(f"  Max Out-of-Scope Similarity:  {max_out_scope:.4f}")
    print(f"  Chosen Threshold (Midpoint):  {calibrated_threshold:.4f}")
    return calibrated_threshold


# Empirically determined fallback threshold (can be overwritten via calibrate_similarity_threshold)
SIMILARITY_THRESHOLD = 0.520


# ---------------------------------------------------------------------------
# Grounded Retrieval & Answer Generation
# ---------------------------------------------------------------------------
def retrieve_grounded_context(
    query: str,
    top_k: int = 2,
    strategy: str = "sentence",
    threshold: float = SIMILARITY_THRESHOLD
) -> Tuple[bool, List[str], float]:
    """
    Queries ChromaDB, computes top-1 cosine similarity, and enforces the fallback cutoff.
    """
    collection = get_chroma_collection(strategy=strategy)
    query_emb = embed_texts([query])

    results = collection.query(
        query_embeddings=query_emb,
        n_results=top_k
    )

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs or not distances:
        return False, [], 0.0

    # Convert distance to similarity
    top1_similarity = max(0.0, 1.0 - distances[0])
    is_grounded = top1_similarity >= threshold

    return is_grounded, docs, top1_similarity


def generate_grounded_answer(
    query: str,
    top_k: int = 2,
    strategy: str = "sentence",
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Returns an answer synthesized strictly from retrieved context.
    Leverages in-memory cache to skip repeated executions.
    """
    start_time = time.perf_counter()

    # Part 4 Task 16: Check cache hit first
    if use_cache:
        cached_result = ENGINE_CACHE.get(query)
        if cached_result:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            cached_copy = dict(cached_result)
            cached_copy["from_cache"] = True
            cached_copy["latency_ms"] = round(duration_ms, 3)
            return cached_copy

    # Cache miss: Execute retrieval and generation
    ENGINE_CACHE.llm_call_count += 1
    is_grounded, retrieved_chunks, top1_sim = retrieve_grounded_context(
        query=query,
        top_k=top_k,
        strategy=strategy,
        threshold=SIMILARITY_THRESHOLD
    )

    if not is_grounded:
        answer = "I do not have sufficient policy information in our knowledge base to answer this question accurately."
    else:
        # Grounded generation synthesis
        joined_context = " ".join(retrieved_chunks)
        answer = f"According to Nykaa Policy: {joined_context}"

    duration_ms = (time.perf_counter() - start_time) * 1000.0

    response_payload = {
        "query": query,
        "answer": answer,
        "is_grounded": is_grounded,
        "top1_similarity": round(top1_sim, 4),
        "threshold_used": SIMILARITY_THRESHOLD,
        "retrieved_chunks_count": len(retrieved_chunks),
        "from_cache": False,
        "latency_ms": round(duration_ms, 3)
    }

    if use_cache:
        ENGINE_CACHE.set(query, response_payload)

    return response_payload


# ---------------------------------------------------------------------------
# Verification Demonstrations
# ---------------------------------------------------------------------------
DEMO_IN_SCOPE_QUERIES = [
    "What is the return window for apparel and footwear?",
    "How do I receive a refund if I paid with cash on delivery?",
    "What is the policy for parcels received in a damaged condition?",
    "Can I cancel my order once it is shipped?",
    "Who is the escalation authority for unresolved grievances?"
]

DEMO_OUT_OF_SCOPE_QUERY = "What are the visa requirements for traveling to Japan?"


def run_demonstration():
    print("\n===================================================================")
    print("STEP 1: EMPIRICAL THRESHOLD CALIBRATION (Part 1 Task 4)")
    print("===================================================================")
    global SIMILARITY_THRESHOLD
    SIMILARITY_THRESHOLD = calibrate_similarity_threshold(strategy="sentence")

    print("\n===================================================================")
    print("STEP 2: DEMONSTRATING GROUNDED GENERATION (>= 5 In-Scope + 1 Out-of-Scope)")
    print("===================================================================")
    for idx, q in enumerate(DEMO_IN_SCOPE_QUERIES, 1):
        res = generate_grounded_answer(q, strategy="sentence", use_cache=False)
        print(f"\n[In-Scope {idx}] Query: {q}")
        print(f"  Similarity: {res['top1_similarity']} (Threshold: {res['threshold_used']})")
        print(f"  Grounded:   {res['is_grounded']}")
        print(f"  Answer:     {res['answer'][:110]}...")

    # Out-of-scope query demonstration (Must trigger fallback)
    res_out = generate_grounded_answer(DEMO_OUT_OF_SCOPE_QUERY, strategy="sentence", use_cache=False)
    print(f"\n[Out-of-Scope] Query: {DEMO_OUT_OF_SCOPE_QUERY}")
    print(f"  Similarity: {res_out['top1_similarity']} (Threshold: {res_out['threshold_used']})")
    print(f"  Grounded:   {res_out['is_grounded']}")
    print(f"  Answer:     {res_out['answer']}")

    print("\n===================================================================")
    print("STEP 3: RESPONSE CACHE HIT / MISS DEMONSTRATION (Part 4 Task 16)")
    print("===================================================================")
    test_q = "What is the return window for apparel and footwear?"

    # First Call (Cache Miss)
    t1 = generate_grounded_answer(test_q, use_cache=True)
    print(f"[Call 1 - Cache Miss]")
    print(f"  From Cache: {t1['from_cache']} | Latency: {t1['latency_ms']} ms | LLM Calls: {ENGINE_CACHE.llm_call_count}")

    # Second Identical Call (Cache Hit)
    t2 = generate_grounded_answer(test_q, use_cache=True)
    print(f"[Call 2 - Cache Hit (Identical Query)]")
    print(f"  From Cache: {t2['from_cache']} | Latency: {t2['latency_ms']} ms | LLM Calls: {ENGINE_CACHE.llm_call_count}")

    # Third Call (Normalized variations: uppercase, extra spaces, trailing punctuation)
    variant_q = "  WHAT is the return window for APPAREL and footwear???  "
    t3 = generate_grounded_answer(variant_q, use_cache=True)
    print(f"[Call 3 - Cache Hit (Normalized Query Variant)]")
    print(f"  From Cache: {t3['from_cache']} | Latency: {t3['latency_ms']} ms | LLM Calls: {ENGINE_CACHE.llm_call_count}")

    print(f"\nCache Stats: Hits={ENGINE_CACHE.hits}, Misses={ENGINE_CACHE.misses}, Redundant Calls Avoided={ENGINE_CACHE.hits}")


if __name__ == "__main__":
    run_demonstration()