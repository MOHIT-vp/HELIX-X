# HELIXX Biomedical Intelligence Backend

This document describes the backend runtime, data flow, API contract, and storage model.

## 1. Purpose

The backend is a local FastAPI service for graph-based compound-disease inference. It loads a typed biomedical graph, creates a heterogeneous graph transformer, requires trained weights, caches graph embeddings, scores pairs, ranks candidates, and returns deterministic graph evidence.

It does not call an LLM, PubMed, an external retrieval service, or any other external API.

## 2. Structure

```text
graphrag-streamlit/
|-- backend/
|   |-- run.py                  # Uvicorn entrypoint
|   `-- app/
|       |-- main.py             # FastAPI app and routes
|       |-- schemas.py          # Pydantic request/response models
|       |-- config.py           # Environment-driven runtime settings
|       |-- check.py            # Backend self-check command
|       `-- service.py          # Loading, inference, ranking, evidence
|-- src/
|   |-- graph_loader.py         # Real graph and synthetic fallback
|   |-- model.py                # HGT model and decoder
|   |-- inference.py            # Embedding, scoring, and ranking
|   `-- explainer.py            # Metapath traversal
|-- data/
|   |-- hetionet_data.pt       # Optional serialized graph
|   `-- node_mappings.json      # Optional node names
|-- assets/
|   `-- model_epoch8_score0.3792.pt  # Optional trained checkpoint
`-- frontend/
    `-- src/api.ts              # Frontend API client
`-- tests/                      # Standard-library backend tests
```

## 3. Running the backend

From `graphrag-streamlit`:

```bash
PYTHONPATH=. .venv/bin/python backend/run.py
```

The service listens on the configured host and port, defaulting to `http://127.0.0.1:8000`. OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

Declared dependencies in `backend/requirements.txt` are:

- `fastapi==0.115.6`
- `uvicorn[standard]==0.34.0`
- `pydantic==2.10.4`

The runtime also requires the PyTorch and PyTorch Geometric packages imported by the source modules.

## 4. Startup lifecycle

`backend/app/main.py` creates one global `InferenceService`. FastAPI's lifespan handler calls `service.load()` before serving requests.

The load sequence is:

1. Environment configuration is loaded and validated.
2. `auto_load()` loads the real graph and mappings, or builds a deterministic synthetic graph only when demo mode permits it.
3. `GraphRAGLinkPredictor` is instantiated on the selected device.
4. A first `encode()` call materializes lazy model parameters.
5. The trained checkpoint is loaded with safe loading and missing/unexpected state-dict keys are rejected.
6. `precompute_embeddings()` runs full-graph message passing once.
7. Embedding node types, counts, dimensions, and memory size are validated.
8. The model switches to evaluation mode and the service becomes ready.

If startup fails, the real exception is logged and stored in `startup_error`, the model and embeddings are cleared, and inference endpoints return HTTP 503. The health state is `starting`, `ready`, `degraded`, or `unavailable`.

Allowed CORS origins are configured through `GRAPHRAG_ALLOWED_ORIGINS`; the defaults support localhost and loopback on ports 5173 and 5174. Wildcard origins are not used.

### Runtime modes

- `production`: requires the real graph, mappings, trained checkpoint, compatible model state, and successful embeddings.
- `demo`: permits the deterministic synthetic graph, but still requires a trained checkpoint before inference is enabled.
- `unavailable`: the graph or checkpoint/model lifecycle failed; entity metadata may remain visible, but prediction and ranking are disabled.

The backend never serves scores from randomly initialized or untrained weights. A missing checkpoint means `model_loaded: false`, `inference_ready: false`, and HTTP 503 for inference routes.

### Configuration

Settings are read by `backend/app/config.py`. Relative paths are resolved from the repository root, regardless of the directory used to launch the process.

| Variable | Default | Purpose |
|---|---|---|
| `GRAPHRAG_MODE` | `auto` | `auto`, `production`, or `demo` |
| `GRAPHRAG_DATA_DIR` | `./data` | Graph data directory |
| `GRAPHRAG_GRAPH_FILE` | `hetionet_data.pt` | Graph filename or path |
| `GRAPHRAG_MAPPINGS_FILE` | `node_mappings.json` | Mapping filename or path |
| `GRAPHRAG_WEIGHTS_PATH` | `./assets/model_epoch8_score0.3792.pt` | Trained checkpoint |
| `GRAPHRAG_ALLOW_DEMO_GRAPH` | `true` | Permit synthetic fallback outside production mode |
| `GRAPHRAG_HOST` | `127.0.0.1` | Bind host |
| `GRAPHRAG_PORT` | `8000` | Bind port |
| `GRAPHRAG_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `GRAPHRAG_LOG_LEVEL` | `INFO` | Python logging level |
| `GRAPHRAG_ALLOWED_ORIGINS` | localhost 5173/5174 | Comma-separated CORS origins |

## 5. Graph and model

### Graph files

The expected real inputs are:

- `data/hetionet_data.pt`: a PyTorch Geometric `HeteroData` graph.
- `data/node_mappings.json`: `{node_type: {index: name}}` mappings.

The graph contains typed biomedical nodes and relationships with 384-dimensional node features. If either real file is unavailable, `build_demo_graph()` creates a deterministic synthetic Hetionet-like graph using seed `42`. Demo data exercises the UI and API but is not scientifically meaningful.

### Model architecture

`GraphRAGLinkPredictor` uses:

- Per-type projection from 384 to 256 dimensions.
- Three Heterogeneous Graph Transformer (`HGTConv`) layers.
- Four attention heads.
- Per-type `LayerNorm` and GELU processing.
- A hybrid decoder combining DistMult, TransE, and dot-product features.
- A refinement MLP with dimensions `3 -> 16 -> 1`.

The decoder scores the `Compound -> treats -> Disease` relation. Scores are sigmoid model scores used for ranking, not calibrated probabilities.

### Cached inference

Full graph message passing runs once at startup. Detached tensors are stored in `InferenceService.embeddings` on the selected device. Requests use these cached embeddings for pair scoring and top-k ranking without repeating full-graph message passing. The service records generation time, embedding shape, device, and memory footprint.

## 6. Evidence generation

`src/explainer.py` traverses fixed metapaths such as:

- Compound -> Gene -> Disease
- Compound -> Gene -> Pathway -> Gene -> Disease

It returns concrete graph paths up to the requested limit. The service converts them into an `EvidenceGraph` containing unique nodes, relationship edges, and paths.

If no short canonical path exists, the response still includes the requested compound and disease connected by an edge labeled `inference target`. This represents the scored request, not an observed biological relationship.

Evidence is deterministic graph traversal, not literature evidence. The `generate_narrative()` helper exists but is not used by the FastAPI routes.

## 7. API

The versioned base URL is `http://127.0.0.1:8000/api/v1`. The previous `/api` paths remain available as backward-compatible aliases, and the frontend uses `/api/v1` by default.

### `GET /api/v1/health` (also `/api/health`)

Returns HTTP 200 with:

```json
{"status": "ready"}
```

Possible values are `starting`, `ready`, `degraded`, and `unavailable`. Health is a service-state endpoint; only `ready` means inference can run.

### `GET /api/v1/status` (also `/api/status`)

Returns:

| Field | Type | Meaning |
|---|---|---|
| `service` | string | Service name |
| `status` | `starting`, `ready`, `degraded`, `unavailable` | Lifecycle state |
| `mode` | `production`, `demo`, `unavailable` | Runtime mode |
| `graph_source` | `hetionet` or `synthetic_demo` | Loaded graph source |
| `graph_label` | string | Human-readable graph name |
| `model_loaded` | boolean | Trained checkpoint loaded |
| `checkpoint` | string | Checkpoint description or `unavailable` |
| `embeddings_cached` | boolean | Startup embedding pass completed |
| `inference_ready` | boolean | Requests can run |
| `node_count` | integer | Total nodes |
| `edge_count` | integer | Total typed edges |
| `feature_dimension` | integer | Normally 384 |
| `device` | string | `cpu` or `cuda` used for inference |
| `api_integrations` | string array | Currently empty |
| `startup_error` | string or null | Diagnostic when startup failed |

### `GET /api/v1/entities/compounds` and `/api/v1/entities/diseases`

The unversioned `/api/entities/...` aliases are also preserved.

Return sorted entity arrays:

```json
{"id": 0, "name": "Example entity", "type": "Compound"}
```

The type is `Disease` for the disease endpoint. These routes return HTTP 503 only when the graph or mappings are unavailable; entity metadata remains available when only the trained checkpoint is missing.

### `POST /api/v1/predict` (also `/api/predict`)

Request:

```json
{"compound_id": 12, "disease_id": 4}
```

Response fields:

| Field | Type | Meaning |
|---|---|---|
| `compound` | Entity | Selected compound |
| `disease` | Entity | Selected disease |
| `score` | number | Sigmoid model score |
| `score_type` | `model_score` | Score semantics |
| `evidence_path_count` | integer | Number of paths returned |
| `evidence_graph` | EvidenceGraph | Nodes, edges, and paths |
| `scientific_notice` | string | Demo/checkpoint/scientific warning |

### `POST /api/v1/candidates/diseases` (also `/api/candidates/diseases`)

Ranks diseases for a compound.

```json
{"entity_id": 12, "limit": 10}
```

`entity_id` must be a non-negative compound index. `limit` defaults to 10 and must be 1 through 50.

Each candidate contains `rank`, an `entity` object, `score`, `score_type`, and `evidence_path_count`. Evidence counts are computed only for the first eight ranked candidates.

### `POST /api/v1/candidates/compounds` (also `/api/candidates/compounds`)

Ranks compounds for a disease. It has the same request and response shape as `/candidates/diseases`; the source entity is a `Disease` and `direction` is `compounds`.

### Evidence aliases

These all return the same `EvidenceGraph`; both `/api/v1` and legacy `/api` prefixes are supported:

- `POST /api/v1/evidence/path`
- `POST /api/v1/evidence/graph`
- `POST /api/v1/evidence`
- `POST /api/v1/graph/subgraph`

Request:

```json
{"compound_id": 12, "disease_id": 4, "max_paths": 8}
```

`max_paths` defaults to 8 and must be 1 through 12.

An `EvidenceGraph` has:

- `nodes`: unique nodes with `id`, `index`, `name`, `type`, and `path_positions`.
- `edges`: directed relationships with `id`, `source`, `target`, `relationship`, and `path_index`.
- `paths`: path ID, hop count, metapath, node IDs, and path nodes.

### `GET /api/v1/methodology` (also `/api/methodology`)

Returns the static model architecture, pipeline stages, explanation method, and scientific caveats used by the frontend methodology view.

## 8. Validation and errors

FastAPI/Pydantic returns HTTP 422 for invalid request bodies:

- `compound_id`, `disease_id`, and `entity_id` must be non-negative integers.
- Candidate `limit` must be 1 through 50.
- Evidence `max_paths` must be 1 through 12.

Inference routes call `require_ready()`. If startup did not produce a validated graph, compatible trained model, and embedding cache, they return HTTP 503 with a structured error:

```json
{"detail": {"error": "inference_unavailable", "message": "The trained checkpoint could not be loaded.", "service_ready": false}}
```

Unknown entity IDs are validated before tensor access and return HTTP 404:

```json
{"error": "entity_not_found", "message": "Disease with id 999999 does not exist.", "service_ready": false}
```

Malformed bodies and invalid bounds return HTTP 422 with a structured validation error. Unexpected server errors return HTTP 500 without a traceback; detailed tracebacks are logged server-side.

## 9. Database and persistence

There is currently no database in this project. The backend does not use PostgreSQL, MySQL, SQLite, MongoDB, an ORM, migrations, a connection pool, user accounts, request history, or saved sessions.

### Persistent files

| File | Purpose | Required? |
|---|---|---|
| `data/hetionet_data.pt` | Real serialized graph | No; synthetic fallback |
| `data/node_mappings.json` | Real node names | No; synthetic fallback |
| `assets/model_epoch8_score0.3792.pt` | Trained weights | Required before inference |

### Runtime memory

`InferenceService` keeps the loaded `HeteroData`, node mappings, model, detached embeddings on the selected device, readiness flags, diagnostics, and any startup error in memory. It does not write predictions, requests, or user state to disk. Restarting the process rebuilds this state from the files.

## 10. Current checkout and configuration caveats

With missing graph files and demo mode enabled, the service reports `graph_source: synthetic_demo` and `mode: demo`. With a missing checkpoint, `model_loaded` is false, `inference_ready` is false, and the service reports `status: unavailable`; initialized weights are never used for inference.

The backend reads the variables documented in `backend/.env.example`, including graph paths, checkpoint path, host, port, device, log level, CORS origins, and demo policy. Relative paths are resolved robustly from the repository root.

For validated inference, provide:

```text
data/hetionet_data.pt
data/node_mappings.json
assets/model_epoch8_score0.3792.pt
```

### Checkpoint recovery audit

The expected checkpoint filename was not present as a regular file and no Git repository or Git LFS metadata was available in this checkout. The ignored `assets/model_epoch8_score0.3792/` directory was a valid extracted PyTorch archive: it contained `data.pkl`, `byteorder`, `version`, and tensor storage files. The payload was reconstructed into `assets/model_epoch8_score0.3792.pt` and validated with:

- `OrderedDict` state dict with 410 tensors.
- Zero missing keys and zero unexpected keys against the materialized `GraphRAGLinkPredictor`.
- Expected 384-input, 256-hidden, 3-layer, 4-head architecture.
- 14,505,395 model parameters loaded.

The real `data/hetionet_data.pt` and `data/node_mappings.json` were not found in the repository or wider Documents workspace. Therefore the restored checkpoint currently runs only against the validated synthetic graph and is reported as `DEMO READY`, not production-ready.

If the generated `.pt` file is absent but the extracted directory remains, `InferenceService` can temporarily repack and load that validated directory payload. The temporary archive is deleted after loading.

## 11. Useful commands

```bash
# Start
cd graphrag-streamlit
PYTHONPATH=. .venv/bin/python backend/run.py

# Health
curl http://127.0.0.1:8000/api/v1/health

# Status
curl http://127.0.0.1:8000/api/v1/status

# Compounds
curl http://127.0.0.1:8000/api/v1/entities/compounds

# Score a pair
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"compound_id": 0, "disease_id": 0}'

# Self-check: allows an honest demo-ready result
PYTHONPATH=. .venv/bin/python -m backend.app.check

# Strict self-check: requires real production artifacts
PYTHONPATH=. .venv/bin/python -m backend.app.check --strict
```

## 12. Verification performed

The hardened backend was compiled and started successfully with the repository virtualenv. The live startup check confirmed:

- Synthetic graph validation succeeds with 11 node types and 48 edge types.
- The recovered trained checkpoint loads with zero missing/unexpected state-dict keys.
- Runtime state is `status: ready`, `mode: demo`, `model_loaded: true`, `embeddings_cached: true`, and `inference_ready: true`.
- Startup logs report CPU device, 14,505,395 parameters, 1,116,160 bytes of embeddings, and approximately 0.045 seconds for embedding generation.
- No random-weight predictions are served; the checkpoint is real, but the graph is synthetic.
- Both versioned `/api/v1/...` and legacy `/api/...` route families are registered.
- The frontend production build succeeds against the updated `/api/v1` contract and explicitly labels unavailable inference separately from demo mode.

The standard-library backend suite passes 19 tests in approximately 1.4 seconds:

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
```

The live endpoint matrix passed health, status, both entity routes, methodology, prediction, both candidate directions, all four evidence aliases, deterministic repeated prediction, invalid entity 404, negative ID 422, invalid limit 422, and invalid `max_paths` 422.

The browser smoke test on `http://127.0.0.1:5174` passed after clearing stale `.next` output: the UI loaded, entities populated from the API, the hypothesis action returned a model score and evidence paths, and the graph view rendered the returned evidence subgraph. The clean frontend server returned HTTP 200 for normal GET requests.

The current checkout cannot report production mode because the real graph and mappings are absent. After supplying `data/hetionet_data.pt` and `data/node_mappings.json`, run the strict self-check and verify that `/api/v1/status` reports `mode: production` and `graph_source: hetionet`.

## 13. Scientific boundaries

The service produces model-generated ranking hypotheses. A high score does not establish clinical efficacy, treatment effectiveness, or causality. Evidence paths describe relationships present in the loaded graph; they do not prove a biological mechanism or replace literature review, expert assessment, or experimental validation.