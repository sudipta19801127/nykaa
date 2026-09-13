from pydantic import BaseModel
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import StructuredMessage
from config.llm_config import get_autogen_ollama_client

class ReviewVerdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str

def build_review_team() -> RoundRobinGroupChat:
    ollama_client = get_autogen_ollama_client()

    compliance_reviewer = AssistantAgent(
        name="Compliance_Reviewer",
        model_client=ollama_client,
        system_message=(
            "You are a Nykaa Compliance Reviewer. Inspect the draft response against the retrieved policy text. "
            "Flag any ungrounded promises or fabricated rules."
        )
    )

    final_editor = AssistantAgent(
        name="Final_Editor",
        model_client=ollama_client,
        system_message="Produce the final JSON verdict adhering to the ReviewVerdict schema.",
        output_content_type=ReviewVerdict
    )

    # Register StructuredMessage[ReviewVerdict] to prevent runtime crashes[cite: 1]
    team = RoundRobinGroupChat(
        participants=[compliance_reviewer, final_editor],
        max_turns=2, # Capstone constraint: use max_turns=2[cite: 1]
        custom_message_types=[StructuredMessage[ReviewVerdict]]
    )
    return team