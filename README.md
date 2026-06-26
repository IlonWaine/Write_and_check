# AI Email Writer & Critic Agent (Ельф-коректор)

An autonomous, multi-agent AI pipeline designed to draft, critique, and refine professional emails. Built using **LangGraph**, **LangChain**, and **FastAPI**, this project demonstrates a production-grade architecture that ensures output quality through a cyclical **Writer-Critic loop** with resilient, cross-provider fallback mechanisms.

---

## Key Features

* **Multi-Agent Workflow:** Orchestrates two specialized LLM nodes (Writer and Critic) running a stateful loop via LangGraph.
* **Structured AI Critique:** Leverages Pydantic validation (`with_structured_output`) to force the Critic agent to return strict structured validation schemas.
* **Bulletproof Fallbacks:** Features cross-provider resilience using LangChain's `.with_fallbacks()`. If OpenRouter hits a global rate limit (`429`), the pipeline seamlessly switches to Google Gemini API on the fly.
* **Modern Web UI:** Features a sleek, asynchronous single-page interface powered by FastAPI and Tailwind CSS.
* **Enterprise Testing & CI/CD:** Complete code coverage with isolated unit and integration mocking tests via `pytest`, fully integrated with **GitHub Actions**.

---

## Architecture Design

The workflow represents a classic cyclical state graph that operates until a stop-condition is achieved or a system recursion safety-valve is reached.

1. **START:** User provides the raw context/instructions for the email.
2. **Writer Node (`process`):** Drafts the email or revises it based on previous feedback.
3. **Critic Node (`evaluate`):** Reviews the message, generating an explicit schema: `approved: bool` and `suggestions: List[str]`.
4. **Router Node (`route`):** Conditional routing. If `approved` is `True`, it routes to `END`. If `False`, it loops back to the **Writer** with specific feedback instructions.

---

## Tech Stack

* **Orchestration:** LangGraph, LangChain Core
* **AI Providers:** OpenRouter (Llama 3.3, Cohere), Google AI Studio (Gemini 1.5 Flash)
* **Backend:** FastAPI, Pydantic v2, Uvicorn
* **Testing:** PyTest, HTTPX (Mocking)
* **CI/CD:** GitHub Actions

---

## Project Structure

```text
├── .github/
│   └── workflows/
│       └── test.yml          # GitHub Actions CI pipeline
├── src/
│   ├── connection.py         # LLM initializations and fallback logic
│   ├── graph_and_ai.py       # LangGraph state machine & node logic
│   ├── schemas.py            # Pydantic schemas and TypedDict states
│   ├── main_api.py           # FastAPI application endpoints
│   └── index.html            # Tailwind CSS single-page frontend
├── test_main_api.py          # Pytest suite with HTTPX/Agent mocking
├── requirements.txt          # Project dependencies
└── .env                      # API Credentials (ignored by git)

## ⚡ Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/email-writer-critic-agent.git](https://github.com/yourusername/email-writer-critic-agent.git)
cd email-writer-critic-agent

### 2. Set up virtual environment
conda create -n agent_one python=3.11
conda activate agent_one
pip install -r requirements.txt

### 3. Set up environment variables
KEY=sk-or-v1-...your-openrouter-key...
GOOGLE_KEY=AIzaSy...your-google-gemini-key...

## Run and test aplication 

### 1. Run project at localhost 
```Run this command at project root (terminal)
python src/main_api.py

### 1. Test project
```Run this command at project root (terminal)
python -m pytest
