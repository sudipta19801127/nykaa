"""
api/app.py
Production-grade FastAPI backend for the Nykaa Domain Support Agent.
Implements:
- POST /ask: Single-turn query answering with guardrails, crew execution, and audit logging.
- POST /add-document: Ingests new policy documents into the active ChromaDB vector index.
- WebSocket /ws/chat: Disconnect-tolerant multi-turn chat stream.
"""

import os
import time
import uuid
from typing import Optional, List, Dict, Any

# Suppress telemetry prior to imports
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from pydantic import BaseModel, Field

from api.logger import structured_logger
from crew.guardrails import validate_input_guardrails, mask_fixed_pii
from crew.tools import check_order_status
from rag.chunking import chunk_sentence_based
from rag.vector_store import get_chroma_collection, embed_texts

app = FastAPI(
    title="Nykaa Domain Support Agent API",
    description="Multi-agent retail customer support backend orchestrated with CrewAI and AutoGen.",
    version="1.0.0"
)

# ---------------------------------------------------------------------------
# Pydantic Request / Response Models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str = Field(..., description="Customer question or order status lookup prompt.")
    session_id: Optional[str] = Field(default="default_session", description="Session identifier for multi-turn tracking.")

class AskResponse(BaseModel):
    trace_id: str
    session_id: str
    sanitized_query: str
    answer: str
    grounded: bool
    escalation_recommended: bool
    latency_ms: float

class AddDocumentRequest(BaseModel):
    doc_id: str = Field(..., description="Unique document ID (e.g., KB-RET-013).")
    topic: str = Field(..., description="Domain topic key.")
    title: str = Field(..., description="Human-readable title.")
    category: str = Field(..., description="Category: Returns, Refunds, Shipping, etc.")
    text: str = Field(..., description="Clean policy text (2-5 sentences).")

class AddDocumentResponse(BaseModel):
    doc_id: str
    chunks_indexed: int
    status: str
    trace_id: str

class ErrorResponse(BaseModel):
    trace_id: str
    error: str


# ---------------------------------------------------------------------------
# Internal Pipeline Helper
# ---------------------------------------------------------------------------

def process_support_pipeline(raw_query: str, session_id: str, trace_id: str) -> Dict[str, Any]:
    """
    Executes input guardrails, retrieves grounding context or order records,
    and returns synthesized answer metrics.
    """
    is_safe, sanitized_query, guard_reason = validate_input_guardrails(raw_query)
    if not is_safe:
        return {
            "answer": f"Request blocked by safety policy: {guard_reason}",
            "grounded": False,
            "escalation_recommended": False,
            "sanitized_query": sanitized_query,
        }

    # Check for order lookup intents (e.g., matching NYK-1001 pattern)
    import re
    order_match = re.search(r"\bNYK-\d{4}\b", sanitized_query, re.IGNORECASE)
    
    if order_match:
        record_id = order_match.group(0).upper()
        order_info = check_order_status(record_id)
        if "error" in order_info:
            ans = f"Order query error: {order_info['error']}"
            escalate = False
        else:
            ans = (
                f"Order {order_info['record_id']} is currently '{order_info['status']}'. "
                f"Value: INR {order_info['order_value_inr']}. Days since creation: {order_info['days_since_created']}. "
                f"Escalation Score: {order_info['escalation_score']}."
            )
            escalate = order_info.get("recommend_escalation", False)
            if escalate:
                ans += " [Escalation Recommended due to shipping delay or threshold age]."
        return {
            "answer": ans,
            "grounded": True,
            "escalation_recommended": escalate,
            "sanitized_query": sanitized_query,
        }

    # Fallback to RAG knowledge base retrieval
    collection = get_chroma_collection(strategy="sentence")
    query_embeddings = embed_texts([sanitized_query])
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=2
    )

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Groundedness threshold check (cosine distance cutoff)
    if not docs or (distances and distances[0] > 0.65):
        return {
            "answer": "I do not have sufficient policy information in our knowledge base to answer this question accurately.",
            "grounded": False,
            "escalation_recommended": True,
            "sanitized_query": sanitized_query,
        }

    synthesized_answer = f"According to Nykaa Policy: {docs[0]}"
    return {
        "answer": synthesized_answer,
        "grounded": True,
        "escalation_recommended": False,
        "sanitized_query": sanitized_query,
    }


# ---------------------------------------------------------------------------
# HTTP Endpoints
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse, responses={400: {"model": ErrorResponse}})
async def ask_endpoint(payload: AskRequest):
    """
    Main query endpoint handling grounded retrieval and order lookups.
    """
    trace_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        pipeline_output = process_support_pipeline(
            raw_query=payload.query,
            session_id=payload.session_id,
            trace_id=trace_id
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Structured JSON-Lines logging with fixed PII masked
        structured_logger.log_event(
            event_type="http_ask",
            trace_id=trace_id,
            request_text=payload.query,
            response_text=pipeline_output["answer"],
            latency_ms=latency_ms,
            status_code=200,
            extra={
                "session_id": payload.session_id,
                "grounded": pipeline_output["grounded"],
                "escalation_recommended": pipeline_output["escalation_recommended"]
            }
        )

        return AskResponse(
            trace_id=trace_id,
            session_id=payload.session_id,
            sanitized_query=pipeline_output["sanitized_query"],
            answer=pipeline_output["answer"],
            grounded=pipeline_output["grounded"],
            escalation_recommended=pipeline_output["escalation_recommended"],
            latency_ms=round(latency_ms, 2)
        )

    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        structured_logger.log_event(
            event_type="http_ask_error",
            trace_id=trace_id,
            request_text=payload.query,
            latency_ms=latency_ms,
            status_code=500,
            extra={"error": str(exc)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"trace_id": trace_id, "error": str(exc)}
        )


@app.post("/add-document", response_model=AddDocumentResponse)
async def add_document_endpoint(payload: AddDocumentRequest):
    """
    Splits, embeds, and indexes a new policy document into ChromaDB.
    """
    trace_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    chunks = chunk_sentence_based(payload.text)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provided text could not be chunked into valid sentences."
        )

    collection = get_chroma_collection(strategy="sentence")
    chunk_ids = [f"{payload.doc_id}_sent_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": payload.doc_id,
            "topic": payload.topic,
            "title": payload.title,
            "category": payload.category,
            "chunk_index": i
        }
        for i in range(len(chunks))
    ]
    embeddings = embed_texts(chunks)

    collection.upsert(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    structured_logger.log_event(
        event_type="add_document",
        trace_id=trace_id,
        request_text=f"Indexed doc {payload.doc_id}",
        latency_ms=latency_ms,
        status_code=200,
        extra={"chunks_count": len(chunks)}
    )

    return AddDocumentResponse(
        doc_id=payload.doc_id,
        chunks_indexed=len(chunks),
        status="Document indexed successfully into ChromaDB.",
        trace_id=trace_id
    )


# ---------------------------------------------------------------------------
# WebSocket Endpoint (Handles client disconnection mid-conversation)
# ---------------------------------------------------------------------------

@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    Real-time multi-turn conversation endpoint.
    Catches WebSocketDisconnect cleanly to ensure the server remains online for all other clients.
    """
    await websocket.accept()
    session_id = f"ws_{uuid.uuid4().hex[:8]}"

    try:
        while True:
            raw_text = await websocket.receive_text()
            turn_trace_id = str(uuid.uuid4())
            start_time = time.perf_counter()

            pipeline_result = process_support_pipeline(
                raw_query=raw_text,
                session_id=session_id,
                trace_id=turn_trace_id
            )

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            structured_logger.log_event(
                event_type="ws_turn",
                trace_id=turn_trace_id,
                request_text=raw_text,
                response_text=pipeline_result["answer"],
                latency_ms=latency_ms,
                status_code=200,
                extra={"session_id": session_id}
            )

            # Send back JSON payload to WebSocket client
            await websocket.send_json({
                "trace_id": turn_trace_id,
                "session_id": session_id,
                "answer": pipeline_result["answer"],
                "grounded": pipeline_result["grounded"],
                "escalation_recommended": pipeline_result["escalation_recommended"]
            })

    except WebSocketDisconnect:
        # Catches abrupt client drop without crashing the uvicorn process
        structured_logger.log_event(
            event_type="ws_client_disconnect",
            trace_id=str(uuid.uuid4()),
            status_code=1000,
            extra={"session_id": session_id, "note": "Client closed connection cleanly."}
        )
    except Exception as exc:
        structured_logger.log_event(
            event_type="ws_server_error",
            trace_id=str(uuid.uuid4()),
            status_code=1011,
            extra={"session_id": session_id, "error": str(exc)}
        )
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)


# ---------------------------------------------------------------------------
# Health / Root Check
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Nykaa Support Agent Backend"}