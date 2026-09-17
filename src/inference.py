"""
Inference wrappers.

Strategy:
  1. Load HeteroData + weights ONCE at startup.
  2. Precompute per-type node embeddings via full message passing (runs ~once).
  3. All user queries hit only the cheap decoder (microseconds).
"""
from __future__ import annotations
import torch
from .model import GraphRAGLinkPredictor


def load_model(weights_path: str, in_channels: int = 384, hidden_channels: int = 256,
               num_heads: int = 4, num_layers: int = 3,
               device: str = "cpu") -> GraphRAGLinkPredictor:
    """Load the trained model from a state_dict .pt file."""
    model = GraphRAGLinkPredictor(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    # PyG HGTConv uses lazy init — we need a dummy forward to materialize
    # the parameter shapes before load_state_dict will accept the file.
    # Usually done by running on the real graph's metadata first.
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


@torch.no_grad()
def precompute_embeddings(model: GraphRAGLinkPredictor, hetero_data,
                          device: str = "cpu") -> dict[str, torch.Tensor]:
    """Run a single full forward pass through the encoder and cache embeddings.
    Called ONCE at app startup (Streamlit @st.cache_resource).
    """
    x_dict = {nt: hetero_data[nt].x.to(device) for nt in hetero_data.node_types}
    edge_index_dict = {
        et: hetero_data[et].edge_index.to(device)
        for et in hetero_data.edge_types
    }
    embeddings = model.encode(x_dict, edge_index_dict)
    return {k: v.detach() for k, v in embeddings.items()}


@torch.no_grad()
def top_k_diseases_for_compound(model, embeddings, compound_idx: int, k: int = 10):
    """Given a compound index, return top-k disease indices and their scores."""
    scores = model.score_all_diseases(embeddings, compound_idx)
    top_vals, top_idx = torch.topk(scores, min(k, scores.numel()))
    return top_idx.tolist(), top_vals.tolist()


@torch.no_grad()
def top_k_compounds_for_disease(model, embeddings, disease_idx: int, k: int = 10):
    """Given a disease index, return top-k compound indices and scores.
    This is the drug-repurposing workflow.
    """
    scores = model.score_all_compounds(embeddings, disease_idx)
    top_vals, top_idx = torch.topk(scores, min(k, scores.numel()))
    return top_idx.tolist(), top_vals.tolist()


@torch.no_grad()
def score_pair(model, embeddings, compound_idx: int, disease_idx: int) -> float:
    return model.score_pair(embeddings, compound_idx, disease_idx)
