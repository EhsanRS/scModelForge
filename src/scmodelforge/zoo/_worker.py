#!/usr/bin/env python
"""Subprocess worker for isolated model embedding extraction.

This script runs inside an isolated virtualenv's Python interpreter.
It does NOT import scModelForge at module level — the source path is
injected at runtime via ``sys.path`` so adapter code can be accessed
without installing the full scModelForge package in the isolated env.

Usage::

    python _worker.py --config /path/to/config.json

Config JSON schema::

    {
        "scmodelforge_src_path": "/path/to/scModelForge/src",
        "adapter_module": "scmodelforge.zoo.geneformer",
        "adapter_class": "GeneformerAdapter",
        "adapter_kwargs": {"model_name_or_path": "ctheodoris/Geneformer", "device": "cpu"},
        "input_h5ad": "/tmp/scmf_iso_xxx/input.h5ad",
        "output_npy": "/tmp/scmf_iso_xxx/output.npy",
        "batch_size": 64,
        "device": "cpu"
    }
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback


def main() -> int:
    """Run the worker: load adapter, extract embeddings, save to numpy.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(description="scModelForge isolated worker")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    try:
        with open(args.config, encoding="utf-8") as f:
            config = json.load(f)

        # Inject scModelForge source path so adapter code is importable
        src_path = config["scmodelforge_src_path"]
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # Import adapter class
        module = importlib.import_module(config["adapter_module"])
        adapter_cls = getattr(module, config["adapter_class"])

        # Instantiate adapter
        adapter_kwargs = config.get("adapter_kwargs", {})
        adapter = adapter_cls(**adapter_kwargs)

        # Load data
        import anndata as ad
        import numpy as np

        adata = ad.read_h5ad(config["input_h5ad"])

        # Extract embeddings
        batch_size = config.get("batch_size", 64)
        device = config.get("device", "cpu")
        embeddings = adapter.extract_embeddings(adata, batch_size=batch_size, device=device)

        # Save output
        np.save(config["output_npy"], embeddings)

        return 0

    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
