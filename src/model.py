"""
GraphRAG link-prediction model architecture.
Reverse-engineered to match the provided state_dict exactly:
  - node_embed.projections.<NodeType>.{weight,bias}          in=384, out=256
  - encoder.convs.{0,1,2}                                    3x HGTConv(256, 256, heads=4)
  - encoder.norms.{0,1,2}.<NodeType>.{weight,bias}           per-type LayerNorm
  - decoder.relation_distmult                                (256,)
  - decoder.relation_transe                                  (256,)
  - decoder.alpha                                            scalar
  - decoder.refine.0 (3->16) and decoder.refine.3 (16->1)    refinement MLP
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


# -- Hetionet schema (matches p_rel keys in the state dict) ---------------- #
NODE_TYPES = [
    "Anatomy", "BiologicalProcess", "CellularComponent", "Compound",
    "Disease", "Gene", "MolecularFunction", "Pathway",
    "PharmacologicClass", "SideEffect", "Symptom",
]

FORWARD_EDGES = [
    ("Anatomy", "downregulates", "Gene"),
    ("Anatomy", "expresses", "Gene"),
    ("Anatomy", "upregulates", "Gene"),
    ("Compound", "binds", "Gene"),
    ("Compound", "causes", "SideEffect"),
    ("Compound", "downregulates", "Gene"),
    ("Compound", "palliates", "Disease"),
    ("Compound", "resembles", "Compound"),
    ("Compound", "treats", "Disease"),
    ("Compound", "upregulates", "Gene"),
    ("Disease", "associates", "Gene"),
    ("Disease", "downregulates", "Gene"),
    ("Disease", "localizes", "Anatomy"),
    ("Disease", "presents", "Symptom"),
    ("Disease", "resembles", "Disease"),
    ("Disease", "upregulates", "Gene"),
    ("Gene", "covaries", "Gene"),
    ("Gene", "interacts", "Gene"),
    ("Gene", "participates", "BiologicalProcess"),
    ("Gene", "participates", "CellularComponent"),
    ("Gene", "participates", "MolecularFunction"),
    ("Gene", "participates", "Pathway"),
    ("Gene", "regulates", "Gene"),
    ("PharmacologicClass", "includes", "Compound"),
]
REVERSE_EDGES = [(dst, f"rev_{rel}", src) for src, rel, dst in FORWARD_EDGES]
ALL_EDGE_TYPES = FORWARD_EDGES + REVERSE_EDGES
METADATA = (NODE_TYPES, ALL_EDGE_TYPES)


# -- Components ------------------------------------------------------------ #
class NodeFeatureEmbed(nn.Module):
    """Per-type Linear projection:  raw_features (384-d) -> hidden (256-d)."""
    def __init__(self, in_channels: int = 384, hidden_channels: int = 256,
                 node_types=NODE_TYPES):
        super().__init__()
        self.projections = nn.ModuleDict({
            nt: nn.Linear(in_channels, hidden_channels) for nt in node_types
        })

    def forward(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {nt: self.projections[nt](x) for nt, x in x_dict.items()}


class HGTEncoder(nn.Module):
    """3 stacked HGTConv layers with per-type LayerNorm between each."""
    def __init__(self, hidden_channels: int = 256, num_heads: int = 4,
                 num_layers: int = 3, metadata=METADATA):
        super().__init__()
        self.convs = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata, heads=num_heads)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(hidden_channels)
                           for nt in metadata[0]})
            for _ in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict):
        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {nt: norm_dict[nt](F.gelu(h)) for nt, h in x_dict.items()}
        return x_dict


class HybridDecoder(nn.Module):
    """Scores Compound -> treats -> Disease.
    Combines DistMult, TransE, and dot-product scores through a 3->16->1 MLP.
    """
    def __init__(self, hidden_channels: int = 256):
        super().__init__()
        self.relation_distmult = nn.Parameter(torch.empty(hidden_channels))
        self.relation_transe = nn.Parameter(torch.empty(hidden_channels))
        self.alpha = nn.Parameter(torch.tensor(0.0))
        # Matches state_dict: refine.0 (Linear 3->16), refine.1, refine.2, refine.3 (Linear 16->1)
        self.refine = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )
        nn.init.normal_(self.relation_distmult, std=0.02)
        nn.init.normal_(self.relation_transe, std=0.02)

    def forward(self, h_drug: torch.Tensor, h_disease: torch.Tensor) -> torch.Tensor:
        # h_drug, h_disease: [N, hidden]
        distmult = (h_drug * self.relation_distmult * h_disease).sum(-1, keepdim=True)
        transe = -torch.norm(
            h_drug + self.relation_transe - h_disease, p=2, dim=-1, keepdim=True
        )
        dot = (h_drug * h_disease).sum(-1, keepdim=True)
        feats = torch.cat([distmult, transe, dot], dim=-1)    # [N, 3]
        return self.refine(feats).squeeze(-1)                 # [N]  -- logits


class GraphRAGLinkPredictor(nn.Module):
    """End-to-end model: HGT encoder + hybrid DistMult/TransE decoder."""
    def __init__(self, in_channels: int = 384, hidden_channels: int = 256,
                 num_heads: int = 4, num_layers: int = 3, metadata=METADATA):
        super().__init__()
        self.node_embed = NodeFeatureEmbed(in_channels, hidden_channels, metadata[0])
        self.encoder = HGTEncoder(hidden_channels, num_heads, num_layers, metadata)
        self.decoder = HybridDecoder(hidden_channels)

    @torch.no_grad()
    def encode(self, x_dict, edge_index_dict):
        """Run full message passing; returns dict of per-type node embeddings."""
        x_dict = self.node_embed(x_dict)
        return self.encoder(x_dict, edge_index_dict)

    @torch.no_grad()
    def score_pair(self, embeddings, compound_idx, disease_idx):
        """Score one (compound, disease) pair using precomputed embeddings."""
        h_c = embeddings["Compound"][compound_idx].unsqueeze(0)
        h_d = embeddings["Disease"][disease_idx].unsqueeze(0)
        return torch.sigmoid(self.decoder(h_c, h_d)).item()

    @torch.no_grad()
    def score_all_diseases(self, embeddings, compound_idx):
        """For a single compound, score against all diseases. Returns [num_diseases]."""
        h_c = embeddings["Compound"][compound_idx].unsqueeze(0)          # [1, H]
        h_d = embeddings["Disease"]                                      # [D, H]
        h_c_exp = h_c.expand_as(h_d)                                     # [D, H]
        logits = self.decoder(h_c_exp, h_d)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def score_all_compounds(self, embeddings, disease_idx):
        """For a single disease, score against all compounds (drug-repurposing)."""
        h_d = embeddings["Disease"][disease_idx].unsqueeze(0)
        h_c = embeddings["Compound"]
        h_d_exp = h_d.expand_as(h_c)
        logits = self.decoder(h_c, h_d_exp)
        return torch.sigmoid(logits)
