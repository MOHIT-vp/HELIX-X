"""
Explanation / "RAG-lite" layer.

Pure graph-traversal explanations — no LLM needed.
For each predicted Compound -> treats -> Disease hypothesis, we find
the top scoring multi-hop paths through the graph (Compound -> Gene -> Disease,
Compound -> Gene -> Pathway -> Disease, etc.) and render them as natural
sentences.

An optional LLM-based narrative wrapper can be plugged in later via
generate_narrative(), which defaults to a deterministic template.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Any


# Canonical metapaths for drug-disease explanation
METAPATHS: list[list[tuple[str, str, str]]] = [
    # Drug -> Gene -> Disease
    [("Compound", "binds", "Gene"),
     ("Gene", "rev_associates", "Disease")],
    [("Compound", "upregulates", "Gene"),
     ("Gene", "rev_downregulates", "Disease")],
    [("Compound", "downregulates", "Gene"),
     ("Gene", "rev_upregulates", "Disease")],

    # Drug -> Gene -> Pathway -> Gene -> Disease (longer, richer)
    [("Compound", "binds", "Gene"),
     ("Gene", "participates", "Pathway"),
     ("Pathway", "rev_participates", "Gene"),
     ("Gene", "rev_associates", "Disease")],

    # Drug -> SideEffect shared w/ Disease symptoms (indirect)
    [("Compound", "palliates", "Disease")],
    [("Compound", "resembles", "Compound"),
     ("Compound", "treats", "Disease")],
]


def _edge_dict(hetero_data) -> dict[tuple[str, str, str], Any]:
    return {et: hetero_data[et].edge_index for et in hetero_data.edge_types}


def find_paths(hetero_data, mappings, compound_idx: int, disease_idx: int,
               max_paths: int = 5) -> list[dict]:
    """Walk canonical metapaths from compound_idx to disease_idx.
    Returns up to max_paths dicts with the node/edge trace.
    """
    edges = _edge_dict(hetero_data)
    results: list[dict] = []

    for mp in METAPATHS:
        # Check that every edge type in the metapath exists
        if not all(et in edges for et in mp):
            continue
        paths = _walk(edges, mp, compound_idx, disease_idx, limit=3)
        for p in paths:
            results.append({"metapath": mp, "nodes": p})
            if len(results) >= max_paths:
                return _render(results, mp_first=False, mappings=mappings)
    return _render(results, mp_first=False, mappings=mappings)


def _walk(edges, metapath, start_idx, end_idx, limit: int = 3) -> list[list[int]]:
    """Enumerate up to `limit` concrete paths from start_idx to end_idx along metapath."""
    # frontier: list of (current_node_idx, path_so_far)
    frontier: list[tuple[int, list[int]]] = [(start_idx, [start_idx])]
    for step, edge_type in enumerate(metapath):
        next_frontier: list[tuple[int, list[int]]] = []
        src, _, dst = edge_type
        ei = edges[edge_type]                       # [2, E]
        src_nodes = ei[0].tolist()
        dst_nodes = ei[1].tolist()
        adj = defaultdict(list)
        for s, d in zip(src_nodes, dst_nodes):
            adj[s].append(d)

        is_last = (step == len(metapath) - 1)
        for cur, path in frontier:
            for nxt in adj.get(cur, []):
                new_path = path + [nxt]
                if is_last:
                    if nxt == end_idx:
                        next_frontier.append((nxt, new_path))
                        if len(next_frontier) >= limit:
                            break
                else:
                    next_frontier.append((nxt, new_path))
            if is_last and len(next_frontier) >= limit:
                break
        frontier = next_frontier
        if not frontier:
            return []
    # frontier now ends at end_idx
    return [p for _, p in frontier[:limit]]


def _render(results: list[dict], mp_first: bool, mappings: dict) -> list[dict]:
    """Convert raw paths into human-readable strings + structured info."""
    rendered = []
    for r in results:
        mp = r["metapath"]
        nodes = r["nodes"]
        # Node-type sequence: src of first edge, dst of each edge
        types = [mp[0][0]] + [et[2] for et in mp]
        parts = []
        for i, (n_idx, n_type) in enumerate(zip(nodes, types)):
            name = mappings.get(n_type, {}).get(n_idx, f"{n_type}[{n_idx}]")
            parts.append(f"{name} ({n_type})")
            if i < len(mp):
                rel = mp[i][1].replace("rev_", "is ").replace("_", " ")
                parts.append(f" —{rel}→ ")
        rendered.append({
            "path_str": "".join(parts),
            "hops": len(mp),
            "metapath": " → ".join(f"{s}-{r}-{d}" for s, r, d in mp),
            "node_indices": nodes,
        })
    return rendered


def generate_narrative(compound_name: str, disease_name: str, score: float,
                       paths: list[dict]) -> str:
    """Deterministic template narrative.  Swap to an LLM call if desired."""
    lines = [
        f"**Hypothesis:** {compound_name} may be relevant for {disease_name}.",
        f"**Predicted plausibility score:** {score:.3f}",
        "",
        "**Supporting multi-hop evidence from Hetionet:**",
    ]
    if not paths:
        lines.append(
            "No short paths were found between this compound and disease in the "
            "knowledge graph. The model's score is driven by learned embeddings "
            "alone — treat with extra caution."
        )
    else:
        for i, p in enumerate(paths, 1):
            lines.append(f"{i}. *{p['metapath']}*  ({p['hops']} hops)")
            lines.append(f"   {p['path_str']}")
    lines.append("")
    lines.append(
        "> *This is a computationally generated hypothesis, not clinical advice. "
        "Any predicted link requires literature validation and expert review.*"
    )
    return "\n".join(lines)
