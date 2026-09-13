"""
crew/agents.py
Agent definitions with RBAC tool assignments satisfying Least Autonomy.
Supports seamless switching between offline MockLLM and local Ollama (qwen2.5:7b).
"""

import os
from crewai import Agent

# Suppress telemetry across crew execution paths
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crew.tools import check_order_status_tool, rag_lookup_tool
from crew.mock_llm import MockLLM

# Determine LLM backend: Default to MockLLM for keyless, zero-network tests
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "true").lower() in ("true", "1", "yes")

if USE_MOCK_LLM:
    LOCAL_LLM = MockLLM()
else:
    # Use local Ollama instance (qwen2.5:7b)
    from langchain_ollama import ChatOllama
    LOCAL_LLM = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.1
    )

# ---------------------------------------------------------------------------
# Agent 1: Policy Retrieval Specialist
# ---------------------------------------------------------------------------
retrieval_agent = Agent(
    role="Policy Retrieval Specialist",
    goal="Retrieve accurate Nykaa return, refund, shipping, and warranty policies.",
    backstory=(
        "You are an expert on Nykaa's retail policies. You only cite official terms "
        "retrieved from the knowledge base and never invent rules."
    ),
    tools=[rag_lookup_tool],
    llm=LOCAL_LLM,
    verbose=False
)

# ---------------------------------------------------------------------------
# Agent 2: Order Status Auditor
# ---------------------------------------------------------------------------
lookup_agent = Agent(
    role="Order Status Auditor",
    goal="Inspect order lifecycle states and compute continuous escalation scores.",
    backstory=(
        "You verify order records against fulfillment databases. You evaluate shipping delays "
        "and order age to recommend ticket escalations."
    ),
    tools=[check_order_status_tool],
    llm=LOCAL_LLM,
    verbose=False
)

# ---------------------------------------------------------------------------
# Agent 3: Customer Support Composer
# ---------------------------------------------------------------------------
response_composer = Agent(
    role="Customer Support Composer",
    goal="Synthesize audit and policy findings into a polite, structured customer response.",
    backstory=(
        "You are the customer-facing voice of Nykaa support. You combine data from "
        "specialists into clear answers without calling tools directly."
    ),
    tools=[],  # Least Autonomy: Pure synthesis, zero tools allowed
    llm=LOCAL_LLM,
    verbose=False
)

if __name__ == "__main__":
    print(f"Loaded agents using LLM mode: {'MockLLM (Offline)' if USE_MOCK_LLM else 'Ollama (Local)'}")
    print(f"Retrieval Agent Tools: {[t.name for t in retrieval_agent.tools]}")
    print(f"Lookup Agent Tools:    {[t.name for t in lookup_agent.tools]}")
    print(f"Composer Agent Tools:  {[t.name for t in response_composer.tools]}")