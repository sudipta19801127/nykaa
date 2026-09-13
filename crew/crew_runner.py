"""
crew/crew_runner.py
Orchestrates the CrewAI multi-agent execution.
Implements:
- Part 2 Task 7: Crew execution with Retrieval, Lookup, and Composer agents.
- Part 2 Task 8: In-memory session tracking via LangChain's InMemoryChatMessageHistory.
- Part 2 Task 9: Pydantic structured output mapping.
- Part 4 Task 15: Per-request budget guardrail.
"""

import os
import json
from typing import Dict, Any
from pydantic import BaseModel, Field

# Suppress telemetry as specified in capstone requirements[cite: 1]
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Crew, Process, Task
from langchain_core.chat_history import InMemoryChatMessageHistory

# Import the pre-configured agents
from crew.agents import retrieval_agent, lookup_agent, response_composer
from governance.budget_guard import check_request_budget

# ---------------------------------------------------------------------------
# Part 2 Task 9: Structured Output Schema
# ---------------------------------------------------------------------------
class FinalCrewResponse(BaseModel):
    answer: str = Field(..., description="The synthesized customer support response.")
    grounded: bool = Field(..., description="True if supported by policy context or valid order data.")
    escalation_recommended: bool = Field(..., description="True if an escalation threshold is met or policy mandates it.")


# ---------------------------------------------------------------------------
# Part 2 Task 8: Session Memory Management
# ---------------------------------------------------------------------------
# In-process memory store maintaining conversation history across turns[cite: 1]
SESSION_STORE: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Retrieves or initializes the chat history for a given session."""
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return SESSION_STORE[session_id]


# ---------------------------------------------------------------------------
# Crew Orchestration
# ---------------------------------------------------------------------------
def execute_support_crew(query: str, session_id: str = "default_session") -> FinalCrewResponse:
    """
    Executes the multi-agent pipeline for a single query.
    Enforces budget constraints, loads memory, runs agents, and updates state.
    """
    
    # Part 4 Task 15: Enforce Runtime Budget Cap (Reject oversized requests)[cite: 1]
    # Throws an exception if the query exceeds the defined token/character limit
    check_request_budget(query)
    
    # Load session state
    history = get_session_history(session_id)
    history_context = ""
    for msg in history.messages:
        prefix = "Customer" if msg.type == "human" else "Agent"
        history_context += f"{prefix}: {msg.content}\n"
    
    if history_context:
        history_context = f"--- Conversation History ---\n{history_context}\n----------------------------\n"

    # Define Agent Tasks (Part 2 Task 7)[cite: 1]
    retrieval_task = Task(
        description=(
            f"{history_context}"
            f"Analyze the current customer query: '{query}'\n"
            "If it asks about policies, warranties, or shipping, use the RAG tool to extract the exact terms. "
            "If it does not ask about policies, state 'No policy retrieval required.'"
        ),
        expected_output="A clean summary of the retrieved policy details, or a note saying none were needed.",
        agent=retrieval_agent
    )

    lookup_task = Task(
        description=(
            f"Review the customer query: '{query}'\n"
            "If there is an order ID (format: NYK-XXXX), use the lookup tool to fetch its status and escalation score. "
            "If no order ID is provided, state 'No order lookup required.'"
        ),
        expected_output="The fetched order metrics and escalation recommendation, or a note saying none were found.",
        agent=lookup_agent
    )

    compose_task = Task(
        description=(
            "Synthesize the findings from the Retrieval Specialist and Lookup Auditor into a single, polite reply to the customer. "
            "Do not hallucinate policies. If neither tool provided useful info, trigger an 'I don't know' fallback response."
        ),
        expected_output="A JSON object matching the FinalCrewResponse schema.",
        agent=response_composer,
        output_pydantic=FinalCrewResponse,  # Enforces Part 2 Task 9 structured output validation[cite: 1]
        context=[retrieval_task, lookup_task] # Explicit dependency graph
    )

    # Assemble the Crew
    support_crew = Crew(
        agents=[retrieval_agent, lookup_agent, response_composer],
        tasks=[retrieval_task, lookup_task, compose_task],
        process=Process.sequential,
        verbose=True
    )

    # Kickoff execution
    result = support_crew.kickoff()

    # Parse output gracefully (Handling standard pydantic attributes or falling back to raw JSON parsing)
    if hasattr(result, "pydantic") and result.pydantic:
        final_response = result.pydantic
    else:
        try:
            # Fallback if the local model drops the markdown fences weirdly
            raw_content = str(result.raw).replace("```json", "").replace("```", "").strip()
            parsed_data = json.loads(raw_content)
            final_response = FinalCrewResponse(**parsed_data)
        except Exception as exc:
            # Failsafe fallback
            final_response = FinalCrewResponse(
                answer=f"Error parsing final response: {str(result.raw)}",
                grounded=False,
                escalation_recommended=True
            )

    # Part 2 Task 8: Update memory state post-execution[cite: 1]
    history.add_user_message(query)
    history.add_ai_message(final_response.answer)

    return final_response


if __name__ == "__main__":
    # Quick Test Execution Block
    print("Testing Crew Execution (RAG + Memory)...")
    
    # Turn 1: RAG question
    res1 = execute_support_crew("What is the return window for Apparel?", session_id="test_1")
    print("\n[Turn 1 Output]")
    print(res1.model_dump_json(indent=2))
    
    # Turn 2: Testing Memory
    res2 = execute_support_crew("And what about Beauty products?", session_id="test_1")
    print("\n[Turn 2 Output (Should utilize memory)]")
    print(res2.model_dump_json(indent=2))
    
    # Turn 3: Lookup Tool Execution
    res3 = execute_support_crew("Can you check status of NYK-1004?", session_id="test_2")
    print("\n[Turn 3 Output (Fresh Session & Tool execution)]")
    print(res3.model_dump_json(indent=2))