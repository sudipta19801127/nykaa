"""
crew/mock_llm.py
Deterministic Mock LLM for CrewAI multi-agent orchestration.
Satisfies the keyless, zero-network requirements of the capstone project.
"""

import os
import re
import json
from typing import Any, List, Optional, Union
from crewai.llms.base_llm import BaseLLM

# Ensure telemetry is strictly disabled prior to execution
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"


class MockLLM(BaseLLM):
    """
    Deterministic BaseLLM subclass that simulates ReAct chain execution
    for Retrieval, Order Lookup, and Response Composition without external APIs.
    """

    def __init__(self, model: str = "mock-deterministic", **kwargs: Any):
        super().__init__(model=model, **kwargs)

    def call(
        self,
        messages: Union[str, List[Any]],
        tools: Optional[List[Any]] = None,
        callbacks: Optional[List[Any]] = None,
        **kwargs: Any
    ) -> str:
        """Executes a single step in CrewAI's ReAct reasoning loop."""
        # Normalize prompt text
        if isinstance(messages, list):
            prompt_parts = []
            for m in messages:
                if isinstance(m, dict):
                    prompt_parts.append(str(m.get("content", "")))
                else:
                    prompt_parts.append(str(getattr(m, "content", m)))
            conversation_text = "\n".join(prompt_parts)
        else:
            conversation_text = str(messages)

        # -------------------------------------------------------------------
        # Pitfall 1 Safe Handling:
        # Check whether an Observation was actually injected following an Action,
        # rather than matching the phrase 'Observation:' in the initial system template.
        # -------------------------------------------------------------------
        has_real_observation = bool(
            re.search(r"Action:[\s\S]*?Action Input:[\s\S]*?Observation:", conversation_text)
        )

        # -------------------------------------------------------------------
        # Agent Role Branch 1: Retrieval Specialist (RAG)
        # -------------------------------------------------------------------
        if "Policy Retrieval Specialist" in conversation_text or "RAG tool" in conversation_text:
            if not has_real_observation:
                # Extract search query from prompt or default to policy question
                query_match = re.search(r"customer query:\s*['\"]?(.*?)['\"]?\n", conversation_text, re.IGNORECASE)
                query_arg = query_match.group(1).strip() if query_match else "Nykaa return and refund policy"
                return (
                    f"Thought: I need to search the knowledge base for relevant policy terms.\n"
                    f"Action: rag_lookup\n"
                    f'Action Input: {{"query": "{query_arg}"}}'
                )
            else:
                # Observation is present; extract text after the last Observation marker
                last_obs = conversation_text.split("Observation:")[-1].strip().split("\n")[0]
                return (
                    f"Thought: I have retrieved the necessary policy context.\n"
                    f"Final Answer: Retrieved policy terms: {last_obs}"
                )

        # -------------------------------------------------------------------
        # Agent Role Branch 2: Order Lookup Auditor
        # -------------------------------------------------------------------
        if "Order Status Auditor" in conversation_text or "lookup tool" in conversation_text:
            order_match = re.search(r"\bNYK-\d{4}\b", conversation_text, re.IGNORECASE)
            
            if not order_match:
                return (
                    "Thought: No order ID was mentioned in the customer query.\n"
                    "Final Answer: No order lookup required as no valid order ID was provided."
                )

            order_id = order_match.group(0).upper()

            if not has_real_observation:
                # Dispatch tool call using structured input schema
                return (
                    f"Thought: An order ID was found. I will look up its status and metrics.\n"
                    f"Action: check_order_status\n"
                    f'Action Input: {{"record_id": "{order_id}"}}'
                )
            else:
                last_obs = conversation_text.split("Observation:")[-1].strip().split("\n")[0]
                return (
                    f"Thought: I have retrieved the order status and escalation metrics.\n"
                    f"Final Answer: Order details verified: {last_obs}"
                )

        # -------------------------------------------------------------------
        # Agent Role Branch 3: Response Composer (Structured Output)
        # -------------------------------------------------------------------
        # Synthesizes inputs into a clean JSON conforming to FinalCrewResponse
        is_grounded = True
        escalate = False

        if "No order lookup required" in conversation_text and "No policy" in conversation_text:
            answer_text = "I do not have sufficient policy information in our knowledge base to answer this question accurately."
            is_grounded = False
        elif "escalation_score" in conversation_text:
            # Parse metrics if present
            answer_text = "Your order details have been reviewed. An update on your delivery status has been retrieved."
            if "recommend_escalation': True" in conversation_text or '"recommend_escalation": true' in conversation_text:
                escalate = True
                answer_text += " Due to shipping timeline delays, this ticket has been flagged for supervisor review."
        else:
            answer_text = (
                "Nykaa provides category-specific return windows. Apparel and Footwear have a 15-day return window, "
                "while Beauty items must be returned sealed within 5 days."
            )

        structured_json = json.dumps({
            "answer": answer_text,
            "grounded": is_grounded,
            "escalation_recommended": escalate
        }, indent=2)

        return structured_json

    async def acall(
        self,
        messages: Union[str, List[Any]],
        tools: Optional[List[Any]] = None,
        callbacks: Optional[List[Any]] = None,
        **kwargs: Any
    ) -> str:
        """Asynchronous execution interface matching synchronous call."""
        return self.call(messages=messages, tools=tools, callbacks=callbacks, **kwargs)

    def supports_function_calling(self) -> bool:
        """Declares tool-calling capability to CrewAI."""
        return True

    def supports_stop_words(self) -> bool:
        return True


if __name__ == "__main__":
    print("--- Testing MockLLM ReAct Loop ---")
    mock = MockLLM()

    # Test 1: Action generation for Policy Retrieval
    rag_prompt = "Policy Retrieval Specialist: customer query: 'What is the return window for Footwear?'"
    step1 = mock.call(rag_prompt)
    print("\n[Step 1 (Agent Action)]:\n" + step1)

    # Test 2: Final Answer after Observation
    step2_input = rag_prompt + "\n" + step1 + "\nObservation: Footwear items are eligible for return within 15 days."
    step2 = mock.call(step2_input)
    print("\n[Step 2 (Agent Final Answer)]:\n" + step2)

    # Test 3: Structured Composer Output
    composer_prompt = "Customer Support Composer: context with Footwear items return within 15 days."
    step3 = mock.call(composer_prompt)
    print("\n[Step 3 (Composer Output)]:\n" + step3)