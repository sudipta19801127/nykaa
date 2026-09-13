"""
governance/least_autonomy.py
Application Layer AI Governance: Principle of Least Autonomy.
Satisfies Part 4 Task 15 by verifying and enforcing that only the authorized
Lookup Agent possesses tool access to `check_order_status` across the crew.
"""

from typing import List, Dict, Any
from crewai import Agent
from crew.tools import check_order_status_tool, rag_lookup_tool


class ToolAuthorizationError(PermissionError):
    """Raised when an agent attempts to execute or bind an unauthorized tool."""
    pass


# Explicit Role-Based Access Control (RBAC) Matrix for Crew Agents
AGENT_TOOL_PERMISSIONS: Dict[str, List[str]] = {
    "Order Status Auditor": ["check_order_status"],
    "Policy Retrieval Specialist": ["rag_lookup"],
    "Customer Support Composer": [],  # Zero tool access allowed (Pure synthesis)
}


def validate_agent_tool_bindings(agent: Agent) -> bool:
    """
    Validates that an agent has only been configured with tools
    explicitly granted under the Least Autonomy policy.
    """
    role = agent.role
    allowed_tools = AGENT_TOOL_PERMISSIONS.get(role, [])

    for tool in getattr(agent, "tools", []) or []:
        tool_name = getattr(tool, "name", str(tool))
        if tool_name not in allowed_tools:
            raise ToolAuthorizationError(
                f"Governance Breach [Application Layer]: Agent '{role}' is not authorized "
                f"to bind or execute tool '{tool_name}'. Allowed tools for this role: {allowed_tools}."
            )
    return True


def verify_crew_autonomy(agents: List[Agent]) -> Dict[str, Any]:
    """
    Scans an entire crew's agent roster to ensure that `check_order_status`
    is strictly isolated to the designated Lookup Agent.
    """
    audit_report = {
        "status": "PASS",
        "violations": [],
        "agent_audits": {}
    }

    for agent in agents:
        agent_tools = [getattr(t, "name", str(t)) for t in (getattr(agent, "tools", []) or [])]
        audit_report["agent_audits"][agent.role] = agent_tools

        # Check order lookup confinement
        if "check_order_status" in agent_tools and agent.role != "Order Status Auditor":
            violation = (
                f"Privilege Escalation Detected: '{agent.role}' holds 'check_order_status' "
                f"in violation of least-autonomy boundaries."
            )
            audit_report["violations"].append(violation)
            audit_report["status"] = "FAIL"

    return audit_report


if __name__ == "__main__":
    print("--- Demonstrating Principle of Least Autonomy (Part 4 Task 15) ---")

    # 1. Instantiate authorized agents
    lookup_agent = Agent(
        role="Order Status Auditor",
        goal="Audit order metrics and calculate escalation risks",
        backstory="Authorized order auditor.",
        tools=[check_order_status_tool],
        verbose=False
    )

    retrieval_agent = Agent(
        role="Policy Retrieval Specialist",
        goal="Retrieve accurate Nykaa policies",
        backstory="Policy specialist.",
        tools=[rag_lookup_tool],
        verbose=False
    )

    composer_agent = Agent(
        role="Customer Support Composer",
        goal="Synthesize responses",
        backstory="Voice of support.",
        tools=[],
        verbose=False
    )

    # Test 1: Valid compliant crew roster
    valid_agents = [retrieval_agent, lookup_agent, composer_agent]
    report = verify_crew_autonomy(valid_agents)
    print("\n[Test 1: Valid Crew Audit]")
    print(f"Audit Status: {report['status']}")
    print(f"Tool Bindings: {report['agent_audits']}")

    # Test 2: Deliberate violation (Wiring check_order_status to Retrieval Agent)
    print("\n[Test 2: Deliberately Violating Least Autonomy]")
    rogue_retrieval_agent = Agent(
        role="Policy Retrieval Specialist",
        goal="Retrieve policies and orders",
        backstory="Rogue agent attempting unauthorized order access.",
        tools=[rag_lookup_tool, check_order_status_tool],
        verbose=False
    )

    try:
        validate_agent_tool_bindings(rogue_retrieval_agent)
        print("Result: Unexpectedly Passed")
    except ToolAuthorizationError as exc:
        print("Result: Successfully Intercepted and Blocked")
        print(f"Caught Security Exception: {exc}")