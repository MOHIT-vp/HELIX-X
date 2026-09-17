from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    graph_file: Path
    mappings_file: Path
    weights_path: Path
    host: str
    port: int
    mode: str
    device: str
    log_level: str
    allowed_origins: tuple[str, ...]
    allow_demo_graph: bool

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        resolved_root = (root or Path(__file__).resolve().parents[2]).resolve()
        data_dir = _resolve_path(os.getenv("GRAPHRAG_DATA_DIR", "./data"), resolved_root)
        graph_file = _resolve_path(os.getenv("GRAPHRAG_GRAPH_FILE", "hetionet_data.pt"), data_dir)
        mappings_file = _resolve_path(os.getenv("GRAPHRAG_MAPPINGS_FILE", "node_mappings.json"), data_dir)
        weights_path = _resolve_path(
            os.getenv("GRAPHRAG_WEIGHTS_PATH", "./assets/model_epoch8_score0.3792.pt"),
            resolved_root,
        )
        origins = tuple(
            origin.strip()
            for origin in os.getenv(
                "GRAPHRAG_ALLOWED_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174",
            ).split(",")
            if origin.strip()
        )
        return cls(
            root=resolved_root,
            data_dir=data_dir,
            graph_file=graph_file,
            mappings_file=mappings_file,
            weights_path=weights_path,
            host=os.getenv("GRAPHRAG_HOST", "127.0.0.1"),
            port=int(os.getenv("GRAPHRAG_PORT", "8000")),
            mode=os.getenv("GRAPHRAG_MODE", "auto").strip().lower(),
            device=os.getenv("GRAPHRAG_DEVICE", "auto").strip().lower(),
            log_level=os.getenv("GRAPHRAG_LOG_LEVEL", "INFO").upper(),
            allowed_origins=origins,
            allow_demo_graph=_env_bool("GRAPHRAG_ALLOW_DEMO_GRAPH", True),
        )

    def validate(self) -> None:
        if self.mode not in {"auto", "production", "demo"}:
            raise ValueError("GRAPHRAG_MODE must be auto, production, or demo")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("GRAPHRAG_DEVICE must be auto, cpu, or cuda")
        if not 1 <= self.port <= 65535:
            raise ValueError("GRAPHRAG_PORT must be between 1 and 65535")


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def select_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("GRAPHRAG_DEVICE=cuda was requested but CUDA is unavailable")
        return "cuda"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"