# GraphRAG API

The FastAPI service loads the graph, model, and cached embeddings once during application startup. It exposes local, keyless inference endpoints for the React frontend.

```bash
cd backend
../.venv/bin/pip install -r requirements.txt
PYTHONPATH=.. ../.venv/bin/python run.py
```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

The service reports synthetic demo mode when `data/hetionet_data.pt` and `data/node_mappings.json` are unavailable. A trained checkpoint is still required before inference runs; the current checkout has a recovered and architecture-validated checkpoint, so it is **DEMO READY**, not production-ready. It does not call an LLM, PubMed, or other external API.
