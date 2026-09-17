from __future__ import annotations

import argparse
import math
from pathlib import Path

from .config import Settings
from .service import InferenceService


def run_check(root: Path, strict: bool = False) -> int:
    settings = Settings.from_env(root)
    checks: list[tuple[str, bool, str]] = []

    try:
        settings.validate()
        checks.append(("Configuration", True, f"mode={settings.mode}, device={settings.device}"))
    except Exception as exc:
        checks.append(("Configuration", False, str(exc)))

    service = InferenceService(root, settings)
    service.load()
    checks.append(("Graph", service.hetero_data is not None, service.status()["graph_label"]))
    checks.append(("Node mappings", bool(service.mappings), f"{len(service.mappings)} node types"))
    checks.append(("Model architecture", service.model is not None or service.startup_error is not None, "constructed or reported failure"))
    checks.append(("Trained checkpoint", service.model_loaded, str(settings.weights_path)))
    checks.append(("Embeddings", service.embeddings_cached, f"{service.embedding_bytes} bytes"))

    inference_ok = False
    evidence_ok = False
    if service.ready:
        try:
            result = service.prediction(0, 0)
            inference_ok = math.isfinite(result["score"])
            graph = result["evidence_graph"]
            node_ids = {node["id"] for node in graph["nodes"]}
            evidence_ok = all(edge["source"] in node_ids and edge["target"] in node_ids for edge in graph["edges"])
        except Exception as exc:
            checks.append(("Inference detail", False, str(exc)))
    checks.append(("Inference", inference_ok, "finite model score"))
    checks.append(("Evidence engine", evidence_ok, "valid node endpoints"))

    status = service.status()
    if strict:
        checks.append(("Production graph", status["graph_source"] == "hetionet", status["graph_label"]))
    for name, passed, detail in checks:
        print(f"[{ 'PASS' if passed else 'FAIL' }] {name}: {detail}")
    print(f"STATUS: {'READY' if service.ready and status['mode'] == 'production' else 'DEMO READY' if service.ready else 'UNAVAILABLE'}")
    if service.startup_error:
        print(f"DIAGNOSTIC: {service.startup_error}")
    required = checks if strict else [check for check in checks if check[0] not in {"Trained checkpoint", "Embeddings", "Inference", "Evidence engine"} or settings.mode == "production"]
    return 0 if all(passed for _, passed, _ in required) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="HELIXX backend self-check")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--strict", action="store_true", help="Require production-ready checkpoint and inference")
    args = parser.parse_args()
    return run_check(args.root.resolve(), args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
