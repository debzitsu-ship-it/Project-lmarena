"""
Export a trained checkpoint (.pt) to a plain NumPy .npz archive so the model
can run WITHOUT PyTorch (e.g. on a phone in Termux, where torch has no wheels).

The .npz contains every weight tensor plus the model config as JSON.
Run:  .venv/bin/python -m my_ai.inference.export_numpy \
          --checkpoint my_ai/checkpoints/latest_release.pt \
          --out my_ai/checkpoints/model_numpy.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="my_ai/checkpoints/latest_release.pt")
    ap.add_argument("--out", default="my_ai/checkpoints/model_numpy.npz")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arrays = {k: v.numpy().astype(np.float32) for k, v in ckpt["model_state"].items()}
    arrays["__config__"] = np.frombuffer(
        json.dumps(ckpt["model_config"]).encode(), dtype=np.uint8)
    np.savez_compressed(args.out, **arrays)
    import os
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB, "
          f"{len(arrays)-1} tensors)")


if __name__ == "__main__":
    main()
