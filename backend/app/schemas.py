from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


ScoreType = Literal["model_score"]


class Entity(BaseModel):
    id: int
    name: str
    type: str


class StatusResponse(BaseModel):
    service: str
    status: Literal["starting", "ready", "degraded", "unavailable"]
    mode: Literal["production", "demo", "unavailable"]
    graph_source: Literal["hetionet", "synthetic_demo"]
    graph_label: str
    model_loaded: bool
    checkpoint: str
    embeddings_cached: bool
    inference_ready: bool
    node_count: int
    edge_count: int
    feature_dimension: int
    device: str
    api_integrations: list[str]
    startup_error: Optional[str] = None


class PredictionRequest(BaseModel):
    compound_id: int = Field(ge=0)
    disease_id: int = Field(ge=0)


class EvidenceNode(BaseModel):
    id: str
    index: int
    name: str
    type: str
    path_positions: list[int] = []


class EvidenceEdge(BaseModel):
    id: str
    source: str
    target: str
    relationship: str
    path_index: int


class EvidencePath(BaseModel):
    id: str
    hops: int
    metapath: str
    node_ids: list[str]
    nodes: list[EvidenceNode]


class EvidenceGraph(BaseModel):
    nodes: list[EvidenceNode]
    edges: list[EvidenceEdge]
    paths: list[EvidencePath]


class PredictionResponse(BaseModel):
    compound: Entity
    disease: Entity
    score: float
    score_type: ScoreType
    evidence_path_count: int
    evidence_graph: EvidenceGraph
    scientific_notice: str


class CandidateRequest(BaseModel):
    entity_id: int = Field(ge=0)
    limit: int = Field(default=10, ge=1, le=50)


class Candidate(BaseModel):
    rank: int
    entity: Entity
    score: float
    score_type: ScoreType
    evidence_path_count: int


class CandidateResponse(BaseModel):
    source: Entity
    direction: Literal["diseases", "compounds"]
    candidates: list[Candidate]
    scientific_notice: str


class EvidenceRequest(PredictionRequest):
    max_paths: int = Field(default=8, ge=1, le=12)


class MethodologyResponse(BaseModel):
    stages: list[dict[str, str]]
    model: dict[str, str]
    explanation: dict[str, str]
    caveats: list[str]
