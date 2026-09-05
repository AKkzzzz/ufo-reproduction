#!/usr/bin/env python3
"""Audit the copied GroundedSAM runtime for Hopper and path portability."""

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_ROOT = REPO_ROOT / "third_party/groundedsam2_env"
HF_CACHE = REPO_ROOT / "third_party/hf_cache"
FORBIDDEN = "/inspire/hdd3/"


def main():
    arch_list = torch.cuda.get_arch_list()
    if "sm_90" not in arch_list and "compute_90" not in arch_list:
        raise RuntimeError(f"GroundedSAM torch lacks Hopper sm_90/PTX support: {arch_list}")

    metadata = [ENV_ROOT / "pyvenv.cfg"]
    site_packages = ENV_ROOT / "lib/python3.10/site-packages"
    metadata.extend(site_packages.glob("*.pth"))
    metadata.extend(site_packages.glob("__editable__*.py"))
    for path in metadata:
        if path.is_file() and FORBIDDEN in path.read_text(errors="ignore"):
            raise RuntimeError(f"copied runtime metadata references {FORBIDDEN}: {path}")

    broken = []
    forbidden_links = []
    for link in HF_CACHE.rglob("*"):
        if not link.is_symlink():
            continue
        try:
            resolved = link.resolve(strict=True)
        except FileNotFoundError:
            broken.append(str(link))
            continue
        if FORBIDDEN in str(resolved):
            forbidden_links.append(f"{link} -> {resolved}")
    if broken or forbidden_links:
        raise RuntimeError(
            f"invalid HF cache symlinks: broken={broken[:5]} forbidden={forbidden_links[:5]}"
        )

    modules = {}
    for name in ("torch", "transformers", "supervision", "sam2"):
        module = __import__(name)
        location = str(Path(module.__file__).resolve())
        if FORBIDDEN in location:
            raise RuntimeError(f"{name} imported from forbidden path: {location}")
        modules[name] = location

    print(f"GROUNDEDSAM_PYTHON={Path(sys.executable).resolve()}")
    print(f"GROUNDEDSAM_TORCH={torch.__version__}")
    print(f"GROUNDEDSAM_TORCH_CUDA={torch.version.cuda}")
    print(f"GROUNDEDSAM_ARCH_LIST={arch_list}")
    for name, location in modules.items():
        print(f"MODULE {name}={location}")
    print("GROUNDEDSAM_RUNTIME_PASS")


if __name__ == "__main__":
    main()
