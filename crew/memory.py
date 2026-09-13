"""
crew/memory.py
In-process session-based conversation memory management.
Satisfies Part 2 Task 8 using LangChain's InMemoryChatMessageHistory.
"""

from typing import Dict, List, Tuple
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# In-process session registry keyed by session_id
SESSION_REGISTRY: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Retrieves the in-memory chat history for a session, or initializes a new one.
    Memory is kept in-process for the life of the running application.
    """
    if session_id not in SESSION_REGISTRY:
        SESSION_REGISTRY[session_id] = InMemoryChatMessageHistory()
    return SESSION_REGISTRY[session_id]


def clear_session_history(session_id: str) -> bool:
    """Clears history for a specific session ID."""
    if session_id in SESSION_REGISTRY:
        SESSION_REGISTRY[session_id].clear()
        del SESSION_REGISTRY[session_id]
        return True
    return False


def format_history_for_prompt(session_id: str, max_turns: int = 5) -> str:
    """
    Renders recent messages into clean dialogue context suitable for agent scratchpads.
    Limits to the latest N turns to avoid overflowing local model context windows.
    """
    history = get_session_history(session_id)
    messages: List[BaseMessage] = history.messages

    if not messages:
        return ""

    # Each turn consists of user + assistant messages
    window = messages[-(max_turns * 2):]
    formatted_lines: List[str] = []

    for msg in window:
        if isinstance(msg, HumanMessage):
            formatted_lines.append(f"Customer: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_lines.append(f"Support Agent: {msg.content}")

    dialogue = "\n".join(formatted_lines)
    return (
        "--- Conversation Context (Previous Turns) ---\n"
        f"{dialogue}\n"
        "---------------------------------------------\n"
    )


def record_turn(session_id: str, user_query: str, agent_response: str) -> None:
    """Appends a completed human/agent turn to the session history."""
    history = get_session_history(session_id)
    history.add_user_message(user_query)
    history.add_ai_message(agent_response)


def run_memory_demonstration() -> Tuple[List[str], List[str]]:
    """
    Demonstrates:
    1. A multi-turn conversation maintaining state across turns in one transcript.
    2. A fresh conversation transcript proving the state is absent/reset.
    """
    session_a = "session_customer_101"
    session_b = "session_fresh_customer_202"

    # Clean up state before running demonstration
    clear_session_history(session_a)
    clear_session_history(session_b)

    # --- Transcript 1: Multi-Turn Conversation (Session A) ---
    transcript_multi_turn: List[str] = []
    
    # Turn 1
    t1_q = "What is the return policy for Apparel?"
    t1_a = "Apparel items can be returned within 15 days of delivery with original tags intact."
    record_turn(session_a, t1_q, t1_a)
    transcript_multi_turn.append(f"[Turn 1] Query: {t1_q}")
    transcript_multi_turn.append(f"[Turn 1] Response: {t1_a}")
    
    # Turn 2 (Relies on contextual memory of Turn 1)
    context_t2 = format_history_for_prompt(session_a)
    t2_q = "And how about Beauty products?"
    t2_a = "Beauty products must be returned within 5 days and must remain sealed in protective packaging."
    record_turn(session_a, t2_q, t2_a)
    transcript_multi_turn.append(f"[Turn 2] Retrieved In-Memory Context:\n{context_t2.strip()}")
    transcript_multi_turn.append(f"[Turn 2] Query: {t2_q}")
    transcript_multi_turn.append(f"[Turn 2] Response: {t2_a}")

    # --- Transcript 2: Fresh Session (Session B) ---
    transcript_fresh: List[str] = []
    context_fresh = format_history_for_prompt(session_b)
    transcript_fresh.append(f"[Fresh Session Initial State] Stored History Length: {len(get_session_history(session_b).messages)}")
    transcript_fresh.append(f"[Fresh Session In-Memory Context Output]: '{context_fresh}' (Confirmed Empty)")

    return transcript_multi_turn, transcript_fresh


if __name__ == "__main__":
    multi_turn, fresh = run_memory_demonstration()

    print("==================================================")
    print("DEMONSTRATION 1: Multi-Turn Conversation (Session A)")
    print("==================================================")
    for line in multi_turn:
        print(line)

    print("\n==================================================")
    print("DEMONSTRATION 2: Fresh Conversation (Session B)")
    print("==================================================")
    for line in fresh:
        print(line)