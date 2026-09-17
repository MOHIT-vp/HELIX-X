from __future__ import annotations

import logging
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import torch

from .config import Settings, select_device

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


class EntityNotFound(ValueError):
    def __init__(self, entity_type: str, index: int):
        super().__init__(f"{entity_type} with id {index} does not exist.")
        self.entity_type = entity_type
        self.index = index


class InferenceService:
    def __init__(self, root: Path = ROOT, settings: Settings | None = None):
        self.root = root.resolve()
        self.settings = settings or Settings.from_env(self.root)
        self.model = None
        self.hetero_data = None
        self.mappings: dict[str, dict[int, str]] = {}
        self.embeddings: dict[str, torch.Tensor] = {}
        self.is_real = False
        self.model_loaded = False
        self.embeddings_cached = False
        self.startup_error: str | None = None
        self.device = "cpu"
        self.embedding_seconds: float | None = None
        self.embedding_bytes = 0
        self.lifecycle = "starting"

    @property
    def ready(self) -> bool:
        return self.model is not None and self.model_loaded and self.embeddings_cached and self.hetero_data is not None

    def load(self) -> None:
        from src.graph_loader import auto_load
        from src.inference import precompute_embeddings
        from src.model import GraphRAGLinkPredictor

        self.lifecycle = "starting"
        self.startup_error = None
        self.model = None
        self.embeddings = {}
        self.model_loaded = False
        self.embeddings_cached = False
        try:
            self.settings.validate()
            self.device = select_device(self.settings.device)
            allow_demo = self.settings.allow_demo_graph and self.settings.mode != "production"
            self.hetero_data, self.mappings, self.is_real = auto_load(
                self.settings.data_dir,
                self.settings.graph_file,
                self.settings.mappings_file,
                allow_demo=allow_demo,
            )
            logger.info("Loaded %s graph with %d node types and %d edge types", "real" if self.is_real else "synthetic demo", len(self.hetero_data.node_types), len(self.hetero_data.edge_types))

            checkpoint_path = self._checkpoint_path()
            if checkpoint_path is None:
                raise FileNotFoundError(f"Trained checkpoint is unavailable: {self.settings.weights_path}")

            self.model = GraphRAGLinkPredictor().to(self.device)
            with torch.inference_mode():
                self.model.encode(
                    {nt: self.hetero_data[nt].x.to(self.device) for nt in self.hetero_data.node_types},
                    {et: self.hetero_data[et].edge_index.to(self.device) for et in self.hetero_data.edge_types},
                )
            state = self._load_checkpoint(checkpoint_path)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"Checkpoint is incompatible (missing keys: {missing}; unexpected keys: {unexpected})")
            self.model_loaded = True
            self.model.eval()

            started = time.perf_counter()
            self.embeddings = precompute_embeddings(self.model, self.hetero_data, device=self.device)
            self.embedding_seconds = time.perf_counter() - started
            self._validate_embeddings()
            self.embeddings_cached = True
            self.lifecycle = "ready"
            parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
            logger.info("MODEL LOADED checkpoint=%s device=%s parameters=%d graph=%s", self.settings.weights_path, self.device, parameter_count, "Hetionet" if self.is_real else "Synthetic Hetionet-like graph")
            logger.info("Cached embeddings on %s in %.3fs (%d bytes)", self.device, self.embedding_seconds, self.embedding_bytes)
        except Exception as exc:
            self.startup_error = str(exc)
            self.lifecycle = "unavailable" if self.model is None or not self.model_loaded else "degraded"
            self.model = None
            self.embeddings = {}
            self.embeddings_cached = False
            logger.exception("Inference service startup failed: %s", exc)

    def _load_checkpoint(self, path: Path) -> dict[str, torch.Tensor]:
        temporary = path.parent == Path(tempfile.gettempdir()) and path.name.startswith("helixx-checkpoint-")
        try:
            try:
                state = torch.load(str(path), map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(str(path), map_location=self.device, weights_only=False)
            if not isinstance(state, dict):
                raise TypeError(f"Checkpoint must contain a state_dict object, got {type(state)}")
            if "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            if not state or not all(isinstance(key, str) for key in state):
                raise TypeError("Checkpoint does not contain a valid state_dict")
            return state
        finally:
            if temporary:
                path.unlink(missing_ok=True)

    def _checkpoint_path(self) -> Path | None:
        if self.settings.weights_path.is_file():
            return self.settings.weights_path
        extracted = self.settings.weights_path.with_suffix("")
        if extracted.is_dir() and (extracted / "data.pkl").is_file():
            return self._pack_extracted_checkpoint(extracted)
        return None

    @staticmethod
    def _pack_extracted_checkpoint(directory: Path) -> Path:
        temporary = tempfile.NamedTemporaryFile(prefix="helixx-checkpoint-", suffix=".pt", delete=False)
        temporary.close()
        with zipfile.ZipFile(temporary.name, "w", compression=zipfile.ZIP_STORED) as archive:
            for source in directory.rglob("*"):
                if source.is_file():
                    archive.write(source, f"{directory.name}/{source.relative_to(directory)}")
        return Path(temporary.name)

    def _validate_embeddings(self) -> None:
        if set(self.embeddings) != set(self.hetero_data.node_types):
            raise RuntimeError("Embedding node types do not match the graph")
        self.embedding_bytes = 0
        for node_type in self.hetero_data.node_types:
            embedding = self.embeddings[node_type]
            if embedding.ndim != 2 or embedding.shape[0] != self.hetero_data[node_type].x.shape[0] or embedding.shape[1] != 256:
                raise RuntimeError(f"Invalid embedding shape for {node_type}: {tuple(embedding.shape)}")
            if not torch.isfinite(embedding).all():
                raise RuntimeError(f"Embeddings for {node_type} contain NaN or Inf values")
            self.embedding_bytes += embedding.numel() * embedding.element_size()

    def _entity(self, entity_type: str, index: int) -> dict[str, Any]:
        if index not in self.mappings.get(entity_type, {}):
            raise EntityNotFound(entity_type, index)
        return {"id": index, "name": self.mappings[entity_type][index], "type": entity_type}

    def validate_entity_id(self, entity_type: str, index: int) -> None:
        self._entity(entity_type, index)

    def entities(self, entity_type: str) -> list[dict[str, Any]]:
        return [self._entity(entity_type, index) for index in sorted(self.mappings.get(entity_type, {}), key=lambda i: self.mappings[entity_type][i].lower())]

    def _paths_and_graph(self, compound_id: int, disease_id: int, max_paths: int = 8) -> tuple[list[dict], dict[str, Any]]:
        from src.explainer import find_paths

        paths = find_paths(self.hetero_data, self.mappings, compound_id, disease_id, max_paths=max_paths)
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        response_paths = []
        for path_index, path in enumerate(paths):
            metapath = path.get("metapath", "").replace(" → ", " -> ")
            relations = [part.split("-", 2) for part in metapath.split(" -> ") if part]
            indices = path.get("node_indices", [])
            path_node_ids: list[str] = []
            path_nodes: list[dict[str, Any]] = []
            for position, index in enumerate(indices):
                node_type = relations[0][0] if position == 0 and relations else (relations[position - 1][2] if position else "Entity")
                node = self._entity(node_type, index)
                node_id = f"{node_type}:{index}"
                path_node_ids.append(node_id)
                if node_id not in nodes:
                    nodes[node_id] = {"id": node_id, "index": index, "name": node["name"], "type": node_type, "path_positions": []}
                nodes[node_id]["path_positions"].append(position)
                path_nodes.append(nodes[node_id])
                if position < len(indices) - 1:
                    relationship = relations[position][1].replace("rev_", "").replace("_", " ") if position < len(relations) else "related"
                    target_id = f"{relations[position][2]}:{indices[position + 1]}"
                    edges.append({"id": f"path-{path_index}-{position}", "source": node_id, "target": target_id, "relationship": relationship, "path_index": path_index})
            response_paths.append({"id": f"path-{path_index}", "hops": path.get("hops", max(0, len(indices) - 1)), "metapath": path.get("metapath", ""), "node_ids": path_node_ids, "nodes": path_nodes})
        if not nodes:
            compound = self._entity("Compound", compound_id)
            disease = self._entity("Disease", disease_id)
            compound_node = {"id": f"Compound:{compound_id}", "index": compound_id, "name": compound["name"], "type": "Compound", "path_positions": [0]}
            disease_node = {"id": f"Disease:{disease_id}", "index": disease_id, "name": disease["name"], "type": "Disease", "path_positions": [1]}
            return paths, {"nodes": [compound_node, disease_node], "edges": [{"id": "inference-target", "source": compound_node["id"], "target": disease_node["id"], "relationship": "inference target", "path_index": -1}], "paths": response_paths}
        return paths, {"nodes": list(nodes.values()), "edges": edges, "paths": response_paths}

    def prediction(self, compound_id: int, disease_id: int, max_paths: int = 8) -> dict[str, Any]:
        self._entity("Compound", compound_id)
        self._entity("Disease", disease_id)
        from src.inference import score_pair

        started = time.perf_counter()
        with torch.inference_mode():
            score = float(score_pair(self.model, self.embeddings, compound_id, disease_id))
        _, graph = self._paths_and_graph(compound_id, disease_id, max_paths)
        logger.info("Pair inference compound=%d disease=%d duration=%.3fs", compound_id, disease_id, time.perf_counter() - started)
        return {"compound": self._entity("Compound", compound_id), "disease": self._entity("Disease", disease_id), "score": score, "score_type": "model_score", "evidence_path_count": len(graph["paths"]), "evidence_graph": graph, "scientific_notice": self.notice}

    def candidates(self, entity_id: int, direction: str, limit: int) -> dict[str, Any]:
        from src.inference import top_k_compounds_for_disease, top_k_diseases_for_compound

        if direction == "diseases":
            self._entity("Compound", entity_id)
            indices, scores = top_k_diseases_for_compound(self.model, self.embeddings, entity_id, k=limit)
            source = self._entity("Compound", entity_id)
            target_type = "Disease"
        else:
            self._entity("Disease", entity_id)
            indices, scores = top_k_compounds_for_disease(self.model, self.embeddings, entity_id, k=limit)
            source = self._entity("Disease", entity_id)
            target_type = "Compound"
        candidates = []
        for rank, (index, score) in enumerate(sorted(zip(indices, scores), key=lambda item: (-float(item[1]), int(item[0]))), 1):
            target = self._entity(target_type, int(index))
            path_count = 0
            if rank <= 8:
                compound_id, disease_id = (entity_id, int(index)) if direction == "diseases" else (int(index), entity_id)
                path_count = len(self._paths_and_graph(compound_id, disease_id, max_paths=5)[1]["paths"])
            candidates.append({"rank": rank, "entity": target, "score": float(score), "score_type": "model_score", "evidence_path_count": path_count})
        return {"source": source, "direction": direction, "candidates": candidates, "scientific_notice": self.notice}

    @property
    def notice(self) -> str:
        if not self.is_real:
            return "Demo mode: synthetic Hetionet-like graph active. Predictions are not scientifically meaningful."
        if not self.model_loaded:
            return "The real graph is loaded, but the trained checkpoint is unavailable. Inference is disabled."
        return "Research prototype: model-generated hypotheses require literature review, expert assessment, and experimental validation."

    def status(self) -> dict[str, Any]:
        node_count = sum(int(self.hetero_data[nt].num_nodes or 0) for nt in self.hetero_data.node_types) if self.hetero_data else 0
        edge_count = sum(int(self.hetero_data[et].edge_index.shape[1]) for et in self.hetero_data.edge_types) if self.hetero_data else 0
        feature_dimension = int(next(iter(self.hetero_data.node_types), "Compound") and self.hetero_data[next(iter(self.hetero_data.node_types))].x.shape[1]) if self.hetero_data else 0
        mode = "production" if self.is_real and self.model_loaded else "demo" if not self.is_real else "unavailable"
        status = "ready" if self.ready else self.lifecycle
        return {"service": "HELIXX Biomedical Intelligence", "status": status, "mode": mode, "graph_source": "hetionet" if self.is_real else "synthetic_demo", "graph_label": "Hetionet" if self.is_real else "Synthetic Hetionet-like graph", "model_loaded": self.model_loaded, "checkpoint": str(self.settings.weights_path) if self.model_loaded else "unavailable", "embeddings_cached": self.embeddings_cached, "inference_ready": self.ready, "node_count": node_count, "edge_count": edge_count, "feature_dimension": feature_dimension, "device": self.device, "api_integrations": [], "startup_error": self.startup_error}
