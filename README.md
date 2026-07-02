# multiagent-automl

Multi-agent AutoML pipeline (CSV-first). See the full roadmap doc for architecture and the day-by-day plan.

## Setup (Day 1)

```bash
# 1. Install uv if you don't have it
pip install uv --break-system-packages   # or: pipx install uv

# 2. Sync the environment (creates .venv, installs everything from pyproject.toml)
uv sync

# 3. Configure API keys
cp .env.example .env
# edit .env: add GOOGLE_API_KEY (aistudio.google.com/apikey) and/or GROQ_API_KEY (console.groq.com/keys)

# 4. Verify the environment is sound
uv run pytest tests/ -v

# 5. Verify your provider keys actually work
uv run python scripts/verify_providers.py
```

## Layout

```
src/automl_agents/
  schemas.py      # PipelineState, EDAReport (Day 1)
  llm_client.py   # provider factory: gemini / groq / ollama (Day 1)
  tools/          # deterministic, unit-tested pipeline functions (Day 2-4)
  nodes/          # LangGraph nodes wrapping tools + LLM calls (Day 5-6)
  graph/          # StateGraph wiring, conditional edges, retry loop (Day 5-7)
tests/            # pytest suite, grows alongside tools/nodes
scripts/          # one-off operational scripts (provider check, dataset prep)
data/raw/         # downloaded datasets (gitignored)
runs/             # per-run parquet snapshots + reports (gitignored)
```
