"""
Graph data loader.

Expects the HeteroData object (saved during training) and a node-name
mappings JSON. If neither is available, builds a small synthetic graph
so the UI can be exercised end-to-end — useful for development only.

Real data files you must provide:
    data/hetionet_data.pt        torch.save(hetero_data, ...)
    data/node_mappings.json      {"Compound": {"0": "Metformin", ...}, ...}
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch_geometric.data import HeteroData
from .model import NODE_TYPES, FORWARD_EDGES, REVERSE_EDGES, ALL_EDGE_TYPES


def load_hetero_data(path: str | Path) -> HeteroData:
    """Load the PyG HeteroData object saved after training."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, HeteroData):
        raise TypeError(
            f"Expected HeteroData at {path}, got {type(obj)}. "
            "Re-export your training graph with torch.save(hetero_data, 'hetionet_data.pt')."
        )
    return obj


def load_node_mappings(path: str | Path) -> dict[str, dict[int, str]]:
    """Load {node_type: {idx: human_name}}.  Keys come back as ints."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a mapping object at {path}")
    mappings: dict[str, dict[int, str]] = {}
    for node_type, values in raw.items():
        if not isinstance(values, dict):
            raise TypeError(f"Mapping for {node_type} must be an object")
        mappings[node_type] = {int(index): str(name) for index, name in values.items()}
    return mappings


def validate_graph(
    data: HeteroData,
    mappings: dict[str, dict[int, str]],
    in_channels: int = 384,
) -> None:
    """Validate the graph contract before model construction or inference."""
    if not isinstance(data, HeteroData):
        raise TypeError(f"Expected HeteroData, got {type(data)}")
    missing_nodes = sorted(set(NODE_TYPES) - set(data.node_types))
    missing_edges = sorted(set(ALL_EDGE_TYPES) - set(data.edge_types))
    if missing_nodes:
        raise ValueError(f"Graph is missing node types: {', '.join(missing_nodes)}")
    if missing_edges:
        raise ValueError(f"Graph is missing edge types: {', '.join(map(str, missing_edges))}")
    missing_mappings = sorted(set(NODE_TYPES) - set(mappings))
    if missing_mappings:
        raise ValueError(f"Mappings are missing node types: {', '.join(missing_mappings)}")

    for node_type in NODE_TYPES:
        store = data[node_type]
        if getattr(store, "x", None) is None or store.x.ndim != 2 or store.x.shape[1] != in_channels:
            shape = getattr(getattr(store, "x", None), "shape", None)
            raise ValueError(f"{node_type} features must have shape [N, {in_channels}], got {shape}")
        count = int(store.x.shape[0])
        expected = set(range(count))
        actual = set(mappings[node_type])
        if actual != expected:
            missing = sorted(expected - actual)[:5]
            extra = sorted(actual - expected)[:5]
            raise ValueError(f"Mappings for {node_type} do not match {count} nodes (missing={missing}, extra={extra})")
        if any(not name.strip() for name in mappings[node_type].values()):
            raise ValueError(f"Mappings for {node_type} contain an empty name")

    for edge_type in ALL_EDGE_TYPES:
        edge_index = data[edge_type].edge_index
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Edge index for {edge_type} must have shape [2, E]")
        source_type, _, target_type = edge_type
        if edge_index.numel() == 0:
            continue
        if edge_index.min().item() < 0 or edge_index[0].max().item() >= data[source_type].x.shape[0] or edge_index[1].max().item() >= data[target_type].x.shape[0]:
            raise ValueError(f"Edge index for {edge_type} contains an out-of-range node ID")


# ---------------------------------------------------------------------- #
#                           DEMO FALLBACK
# ---------------------------------------------------------------------- #
def build_demo_graph(seed: int = 42, in_channels: int = 384) -> tuple[HeteroData, dict]:
    """Tiny synthetic Hetionet for UI testing when real data is unavailable."""
    torch.manual_seed(seed)

    # Small counts so it fits comfortably in memory
    counts = {
        "Anatomy": 50, "BiologicalProcess": 100, "CellularComponent": 30,
        "Compound": 200, "Disease": 80, "Gene": 400,
        "MolecularFunction": 40, "Pathway": 60,
        "PharmacologicClass": 20, "SideEffect": 50, "Symptom": 60,
    }

    data = HeteroData()
    for nt, n in counts.items():
        data[nt].x = torch.randn(n, in_channels)
        data[nt].num_nodes = n

    def rand_edges(n_src, n_dst, n_edges):
        src = torch.randint(0, n_src, (n_edges,))
        dst = torch.randint(0, n_dst, (n_edges,))
        return torch.stack([src, dst], dim=0)

    for src_t, rel, dst_t in FORWARD_EDGES:
        ei = rand_edges(counts[src_t], counts[dst_t], n_edges=500)
        data[(src_t, rel, dst_t)].edge_index = ei

    for dst_t, rev_rel, src_t in REVERSE_EDGES:
        # rev edges are transposes of forwards
        fwd_rel = rev_rel.replace("rev_", "")
        fwd_ei = data[(src_t, fwd_rel, dst_t)].edge_index
        data[(dst_t, rev_rel, src_t)].edge_index = fwd_ei.flip(0)

    # Fake human-readable names
    mappings = {}
    for nt, n in counts.items():
        mappings[nt] = {i: f"{nt}_{i:04d}" for i in range(n)}
    # Nicer names for compounds + diseases
    mappings["Compound"].update({
        0: "Metformin", 1: "Aspirin", 2: "Ibuprofen", 3: "Atorvastatin",
        4: "Omeprazole", 5: "Amoxicillin", 6: "Lisinopril", 7: "Simvastatin",
    })
    mappings["Disease"].update({
        0: "Alzheimer's disease", 1: "Type 2 diabetes", 2: "Hypertension",
        3: "Parkinson's disease", 4: "Breast cancer", 5: "Asthma",
        6: "Depression", 7: "Rheumatoid arthritis",
    })
    return data, mappings


def auto_load(
    data_dir: str | Path = "data",
    graph_path: str | Path | None = None,
    mappings_path: str | Path | None = None,
    allow_demo: bool = True,
) -> tuple[HeteroData, dict, bool]:
    """Load validated real artifacts or an explicitly permitted demo graph.
    Returns (hetero_data, mappings, is_real).
    """
    p = Path(data_dir)
    graph_p = Path(graph_path) if graph_path else p / "hetionet_data.pt"
    map_p = Path(mappings_path) if mappings_path else p / "node_mappings.json"
    if graph_p.exists() and map_p.exists():
        data, mappings = load_hetero_data(graph_p), load_node_mappings(map_p)
        validate_graph(data, mappings)
        return data, mappings, True
    if not allow_demo:
        missing = [str(path) for path in (graph_p, map_p) if not path.exists()]
        raise FileNotFoundError(f"Required graph artefacts are missing: {', '.join(missing)}")
    data, mappings = build_demo_graph()
    validate_graph(data, mappings)
    return data, mappings, False
