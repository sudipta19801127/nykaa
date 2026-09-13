# Nykaa Domain Support Agent (CrewAI + AutoGen + FastAPI)

**Track:** E-commerce & Retail (Nykaa)  
**Deliverable:** Unified End-to-End Autonomous Agent Architecture (Parts 1–4)

---

## 1. Project Overview & Zero-Network Compliance

This repository implements a production-minded retail customer support system for Nykaa. The architecture integrates:
* **Deterministic Synthetic Dataset & Local Dual RAG Engine** (SentenceTransformers + ChromaDB).
* **Multi-Agent Orchestration** via CrewAI (Retrieval, Lookup, and Response Composition).
* **Multi-Turn Session Memory** powered by LangChain's in-process message stores.
* **Secondary Review & Governance Layer** via AutoGen v0.4 (`RoundRobinGroupChat` with structured verdicts).
* **Application & Runtime Governance** (Least Autonomy enforcement and token/cost budget gates).
* **Asynchronous FastAPI Deployment** supporting HTTP endpoints and a resilient WebSocket chat interface with ELK-style structured logging.

### Zero-Network / Keyless Verification
- All language-model operations run without external API keys or paid third-party endpoints.
- **Mock Mode:** Default execution uses an internal, deterministic `MockLLM` subclass extending `crewai.llms.base_llm.BaseLLM`.
- **Local Model Execution:** Validated using local **Ollama** (`qwen2.5:7b`) with `base_url="http://localhost:11434"`.
- **Telemetry Suppression:** Telemetry is strictly suppressed across all execution pathways:
  ```bash
  export CREWAI_DISABLE_TELEMETRY=true
  export OTEL_SDK_DISABLED=true

---

# 1. Generate the deterministic 45-order dataset (Part 1 Task 1)
python -m data.dataset

# 2. Build and embed both ChromaDB collections (Fixed & Sentence) (Part 1 Task 3)
python -m rag.vector_store

python -m rag.evaluate_rag

# Grounded generation, threshold calibration, and cache hits (Part 1 Task 4 & Part 4 Task 16)
python -m rag.grounded_engine

# Multi-turn memory transcript vs. clean session isolation (Part 2 Task 8)
python -m crew.memory

# Principle of Least Autonomy RBAC verification (Part 4 Task 15)
python -m governance.least_autonomy

# Token and capacity budget limit verification (Part 4 Task 15)
python -m governance.budget_guard

uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload