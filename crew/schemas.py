"""
crew/schemas.py
Pydantic schemas and structured output contracts.
Satisfies Part 2 Task 9 (structured output schema) and Part 4 Task 14 (review verdict).
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Part 2 Task 9: Crew Final Output Schema
# ---------------------------------------------------------------------------

class FinalCrewResponse(BaseModel):
    """
    Standard schema to which every customer-facing Crew response must conform.
    Used with CrewAI's output_pydantic validation.
    """
    answer: str = Field(
        ...,
        description="The customer-facing response text synthesized by the composer."
    )
    grounded: bool = Field(
        ...,
        description="True if the answer is grounded in retrieved policies or verified order records."
    )
    escalation_recommended: bool = Field(
        ...,
        description="True if an internal support ticket should be escalated to Level 2/3."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional auxiliary metrics (e.g., order_id, escalation_score, doc_ids)."
    )


# ---------------------------------------------------------------------------
# Part 4 Task 14: AutoGen Review Stage Verdict Schema
# ---------------------------------------------------------------------------

class ReviewVerdict(BaseModel):
    """
    Output model emitted by the AutoGen Final-Editor agent.
    Must contain approved, final_answer, and reason fields as specified.
    """
    approved: bool = Field(
        ...,
        description="True if the draft answer is policy-compliant, accurate, and safe."
    )
    final_answer: str = Field(
        ...,
        description="The final customer-ready text (either unmodified or revised)."
    )
    reason: str = Field(
        ...,
        description="A concise justification describing why the draft was approved or revised."
    )


# ---------------------------------------------------------------------------
# Tool Input / Output Schemas
# ---------------------------------------------------------------------------

class OrderStatusQuery(BaseModel):
    """Input payload for checking an order status."""
    record_id: str = Field(
        ...,
        description="Unique order record ID in the format NYK-XXXX (e.g., NYK-1002)."
    )


class OrderStatusResult(BaseModel):
    """Structured result returned by the check_order_status tool."""
    record_id: str
    status: str
    order_value_inr: int
    days_since_created: int
    delayed_shipment: bool
    escalation_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Continuous score in [0, 1] combining delay penalty and recency."
    )
    recommend_escalation: bool


class RAGQueryInput(BaseModel):
    """Input payload for RAG policy retrieval."""
    query: str = Field(..., description="Natural language policy question to retrieve chunks for.")
    top_k: int = Field(default=2, ge=1, le=5, description="Number of context chunks to retrieve.")


class RAGQueryResult(BaseModel):
    """Structured result returned by the RAG retrieval tool."""
    query: str
    retrieved_chunks: list[str]
    similarity_score: float
    is_grounded: bool