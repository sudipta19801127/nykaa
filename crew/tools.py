"""
crew/tools.py
Domain tools for CrewAI multi-agent orchestration.
Satisfies Part 2 Task 6:
- Continuous escalation score formula:
  escalation_score = 0.60 * delayed_shipment + 0.40 * (days_since_created / 30)
- Order status lookup against synthetic dataset with escalation threshold evaluation.
- RAG retrieval tool bound to ChromaDB sentence collection.
"""

import json
from typing import Dict, Any, Union
from crewai.tools import tool

from data.dataset import get_order_by_id
from rag.vector_store import get_chroma_collection, embed_texts

ESCALATION_CUTOFF = 0.70


def compute_escalation_score(delayed_shipment: bool, days_since_created: int) -> float:
    """
    Computes a continuous escalation risk score in [0.0, 1.0].
    Weights: 60% delay factor, 40% normalized age factor (capped at 30 days).
    """
    delay_component = 0.60 if delayed_shipment else 0.0
    recency_component = 0.40 * min(1.0, max(0.0, days_since_created / 30.0))
    return round(delay_component + recency_component, 4)


def check_order_status(record_id: str) -> Dict[str, Any]:
    """
    Python callable for order verification and escalation risk calculation.
    """
    cleaned_id = record_id.strip().upper()
    order = get_order_by_id(cleaned_id)

    if not order:
        return {
            "error": f"Order '{cleaned_id}' not found in Nykaa system.",
            "record_id": cleaned_id,
            "recommend_escalation": False
        }

    delayed = bool(order.get("delayed_shipment", False))
    days = int(order.get("days_since_created", 0))
    score = compute_escalation_score(delayed, days)
    recommend_escalate = score >= ESCALATION_CUTOFF

    return {
        "record_id": order["record_id"],
        "customer_name": order.get("customer_name"),
        "product_category": order.get("product_category"),
        "order_value_inr": order.get("order_value_inr"),
        "status": order.get("status"),
        "days_since_created": days,
        "delayed_shipment": delayed,
        "escalation_score": score,
        "recommend_escalation": recommend_escalate
    }


# ---------------------------------------------------------------------------
# CrewAI Decorated Tools
# ---------------------------------------------------------------------------

@tool("check_order_status")
def check_order_status_tool(record_id: str) -> str:
    """
    Looks up status, value, age, and escalation risk for an order by ID (e.g., NYK-1002).
    Authorized strictly for the Order Status Auditor.
    """
    result = check_order_status(record_id)
    return json.dumps(result, ensure_ascii=False)


@tool("rag_lookup")
def rag_lookup_tool(query: str) -> str:
    """
    Retrieves relevant Nykaa retail policies, warranties, and SLAs from the knowledge base.
    Authorized strictly for the Policy Retrieval Specialist.
    """
    try:
        collection = get_chroma_collection(strategy="sentence")
        q_emb = embed_texts([query])
        res = collection.query(query_embeddings=q_emb, n_results=2)
        
        docs = res.get("documents", [[]])[0]
        if not docs:
            return "No matching policy terms found."
        return " | ".join(docs)
    except Exception as exc:
        return f"RAG retrieval fallback: unable to query index ({str(exc)})"


if __name__ == "__main__":
    print("--- Testing crew/tools.py ---")
    # Quick sanity test on calculation
    sample_score = compute_escalation_score(delayed_shipment=True, days_since_created=10)
    print(f"Delay=True, Days=10 -> Score: {sample_score} (Escalate: {sample_score >= ESCALATION_CUTOFF})")
    
    # Test tool schema inspectability
    print(f"check_order_status_tool name: {check_order_status_tool.name}")
    print(f"rag_lookup_tool name: {rag_lookup_tool.name}")