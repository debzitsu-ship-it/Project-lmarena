"""
Hardware inspection + automatic model-config recommendation.

We NEVER assume a powerful GPU exists. We look at what the machine
actually has (CPU cores, RAM, GPU/VRAM, CUDA, disk) and recommend the
largest model configuration that can realistically be trained on it.

Run:  python -m my_ai.hardware
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, asdict


@dataclass
class HardwareReport:
    cpu_cores: int
    ram_gb: float
    disk_free_gb: float
    cuda_available: bool
    gpu_name: str | None
    vram_gb: float | None
    mps_available: bool  # Apple Silicon

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def inspect_hardware() -> HardwareReport:
    import torch

    cpu_cores = os.cpu_count() or 1

    # RAM: read from /proc/meminfo on Linux, fall back to psutil-free heuristics.
    ram_gb = 0.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    ram_gb = int(line.split()[1]) / (1024**2)
                    break
    except FileNotFoundError:
        try:
            ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
        except (ValueError, OSError):
            ram_gb = 8.0  # conservative guess

    disk_free_gb = shutil.disk_usage(os.getcwd()).free / (1024**3)

    cuda = torch.cuda.is_available()
    gpu_name, vram_gb = None, None
    if cuda:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        vram_gb = props.total_memory / (1024**3)

    mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    return HardwareReport(cpu_cores, round(ram_gb, 2), round(disk_free_gb, 2),
                          cuda, gpu_name, round(vram_gb, 2) if vram_gb else None, mps)


def recommend_config(hw: HardwareReport) -> tuple[str, str]:
    """Return (config_name, human explanation)."""
    if hw.cuda_available and hw.vram_gb:
        if hw.vram_gb >= 40:
            return "large_300m", f"GPU '{hw.gpu_name}' with {hw.vram_gb}GB VRAM can train ~300M params."
        if hw.vram_gb >= 20:
            return "medium_100m", f"GPU '{hw.gpu_name}' with {hw.vram_gb}GB VRAM can train ~100M params."
        if hw.vram_gb >= 10:
            return "base_50m", f"GPU '{hw.gpu_name}' with {hw.vram_gb}GB VRAM fits the ~50M config."
        if hw.vram_gb >= 6:
            return "small_20m", f"GPU '{hw.gpu_name}' with {hw.vram_gb}GB VRAM fits the ~20M config."
        return "tiny_5m", f"GPU '{hw.gpu_name}' has only {hw.vram_gb}GB VRAM; use the tiny config."
    if hw.mps_available:
        if hw.ram_gb >= 32:
            return "base_50m", "Apple Silicon with >=32GB unified memory fits the ~50M config."
        return "small_20m", "Apple Silicon detected; the ~20M config trains at usable speed."
    # CPU only
    if hw.ram_gb >= 16 and hw.cpu_cores >= 8:
        return "small_20m", "CPU-only but beefy; ~20M params is slow-but-possible."
    return "tiny_5m", (f"CPU-only machine ({hw.cpu_cores} cores, {hw.ram_gb}GB RAM). "
                       "Training is compute-bound: only the tiny (~5M param) config is realistic.")


def main() -> None:
    hw = inspect_hardware()
    print("=== Hardware report ===")
    print(hw.to_json())
    name, why = recommend_config(hw)
    print("\n=== Recommendation ===")
    print(f"config : my_ai/configs/{name}.json")
    print(f"reason : {why}")


if __name__ == "__main__":
    main()
