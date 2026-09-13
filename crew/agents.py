import os
from crewai import Agent
from crew.tools import check_order_status_tool, rag_lookup_tool
from config.llm_config import get_crewai_ollama_llm

# Must disable telemetry per capstone instructions
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

llm = get_crewai_ollama_llm()

retrieval_agent = Agent(
    role="Policy Retrieval Specialist",
    goal="Retrieve accurate Nykaa policies matching user inquiries",
    backstory="You are an expert on Nykaa's terms, return policies, and SLAs.",
    tools=[rag_lookup_tool],
    llm=llm,
    verbose=True
)

lookup_agent = Agent(
    role="Order Status Auditor",
    goal="Look up order metrics and calculate escalation risks",
    backstory="You strictly audit order IDs using internal records.",
    tools=[check_order_status_tool],  # Only lookup agent has access (Part 4 Least Autonomy)
    llm=llm,
    verbose=True
)

response_composer = Agent(
    role="Customer Support Composer",
    goal="Synthesize policy and order facts into a single polite customer reply",
    backstory="You are the final voice of Nykaa support. You do not make tool calls.",
    tools=[],
    llm=llm,
    verbose=True
)