from __future__ import annotations

import logging
import logging.config
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings
from .schemas import (
    CandidateRequest,
    CandidateResponse,
    Entity,
    EvidenceGraph,
    EvidenceRequest,
    MethodologyResponse,
    PredictionRequest,
    PredictionResponse,
    StatusResponse,
)
from .service import EntityNotFound, InferenceService

settings = Settings.from_env(Path(__file__).resolve().parents[2])
logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
service = InferenceService(settings.root, settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting HELIXX backend")
    service.load()
    yield
    logger.info("Stopping HELIXX backend")


app = FastAPI(title="HELIXX Biomedical Intelligence API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(EntityNotFound)
def entity_not_found(_: Request, exc: EntityNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": "entity_not_found", "message": str(exc), "service_ready": service.ready})


@app.exception_handler(RequestValidationError)
def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "validation_error", "message": "Request validation failed.", "details": exc.errors()})


@app.exception_handler(Exception)
def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled backend error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "internal_error", "message": "The backend could not complete the request.", "service_ready": service.ready})


@app.middleware("http")
async def request_logging(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    logger.info("%s %s -> %d (%.3fs)", request.method, request.url.path, response.status_code, time.perf_counter() - started)
    return response


def require_graph() -> None:
    if service.hetero_data is None or not service.mappings:
        raise HTTPException(status_code=503, detail="Graph data is unavailable. Check graph artefacts.")


def require_ready() -> None:
    if not service.ready:
        raise HTTPException(status_code=503, detail={"error": "inference_unavailable", "message": service.startup_error or "The trained checkpoint could not be loaded.", "service_ready": False})


def methodology() -> dict:
    return {"stages": [{"title": "Hetionet", "detail": "Typed biomedical entities and relations"}, {"title": "Heterogeneous graph", "detail": "384-dimensional node features"}, {"title": "3 HGT layers", "detail": "256 hidden channels with 4 attention heads"}, {"title": "Cached embeddings", "detail": "Full message passing once at startup"}, {"title": "Hybrid decoder", "detail": "DistMult + TransE + dot product"}, {"title": "Compound-disease score", "detail": "Model score for Compound -> treats -> Disease"}, {"title": "Path explanation", "detail": "Deterministic metapath traversal; no LLM or PubMed retrieval"}], "model": {"architecture": "3-layer Heterogeneous Graph Transformer", "feature_dimension": "384", "hidden_dimension": "256", "heads": "4", "decoder": "DistMult + TransE + dot-product features -> 3 to 16 to 1 MLP"}, "explanation": {"type": "Metapath-based graph traversal", "llm": "Not used", "literature": "Not used"}, "caveats": ["Scores are ranking outputs, not calibrated probabilities.", "Synthetic fallback predictions are not scientifically meaningful.", "The decoder scores the treats relation only."]}


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ready" if service.ready else service.lifecycle}


@router.get("/status", response_model=StatusResponse)
def status() -> dict:
    return service.status()


@router.get("/entities/compounds", response_model=list[Entity])
def compounds() -> list[dict]:
    require_graph()
    return service.entities("Compound")


@router.get("/entities/diseases", response_model=list[Entity])
def diseases() -> list[dict]:
    require_graph()
    return service.entities("Disease")


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict:
    require_ready()
    return service.prediction(request.compound_id, request.disease_id)


@router.post("/candidates/diseases", response_model=CandidateResponse)
def candidate_diseases(request: CandidateRequest) -> dict:
    require_ready()
    return service.candidates(request.entity_id, "diseases", request.limit)


@router.post("/candidates/compounds", response_model=CandidateResponse)
def candidate_compounds(request: CandidateRequest) -> dict:
    require_ready()
    return service.candidates(request.entity_id, "compounds", request.limit)


@router.post("/evidence/path", response_model=EvidenceGraph)
def evidence_path(request: EvidenceRequest) -> dict:
    require_ready()
    service.validate_entity_id("Compound", request.compound_id)
    service.validate_entity_id("Disease", request.disease_id)
    return service.prediction(request.compound_id, request.disease_id, request.max_paths)["evidence_graph"]


@router.post("/evidence/graph", response_model=EvidenceGraph)
def evidence_graph(request: EvidenceRequest) -> dict:
    return evidence_path(request)


@router.post("/evidence", response_model=EvidenceGraph)
def evidence(request: EvidenceRequest) -> dict:
    return evidence_path(request)


@router.post("/graph/subgraph", response_model=EvidenceGraph)
def graph_subgraph(request: EvidenceRequest) -> dict:
    return evidence_path(request)


@router.get("/methodology", response_model=MethodologyResponse)
def get_methodology() -> dict:
    return methodology()


app.include_router(router, prefix="/api/v1")
app.include_router(router, prefix="/api")
