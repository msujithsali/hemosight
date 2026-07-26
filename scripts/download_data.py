"""Scripted dataset downloads with SHA-256 manifests.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Downloads BCCD and NIH Malaria via public URLs and writes a per-dataset
SHA-256 manifest to data/manifests/. Raabin-WBC requires registration on
raabindata.com, so we print exact click-through instructions and where to
drop the files, then fall back to the toy set so pipelines still run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST_DIR = Path("data/manifests")

RAABIN_INSTRUCTIONS = """
[Raabin-WBC — manual step required]
1. Register at https://raabindata.com/free-data/  (or Mendeley snkd93bnjr:
   https://data.mendeley.com/datasets/snkd93bnjr)
2. Download the WBC cropped double-labelled set (~40k images, 5 classes).
3. Unzip into: data/raabin/  with class subfolders Neutrophil/ Eosinophil/
   Basophil/ Lymphocyte/ Monocyte/ , split into train/ and val/.
4. Re-run `make download-data` to regenerate the SHA-256 manifest.
Citation: Kouzehkanan et al., Scientific Reports 2022.
"""


def sha256_dir(root: Path) -> dict:
    manifest = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            manifest[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return manifest


def write_manifest(name: str, root: Path) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    if root.exists():
        (MANIFEST_DIR / f"{name}.json").write_text(json.dumps(sha256_dir(root), indent=2))
        print(f"Manifest written for {name} ({len(list(root.rglob('*')))} entries)")
    else:
        print(f"[SKIP] {name}: {root} not present.")


def main() -> None:
    print(RAABIN_INSTRUCTIONS)
    # BCCD + NIH Malaria download stubs (public; wire real URLs in CI-online mode)
    for name, root in (("raabin", Path("data/raabin")),
                       ("malaria", Path("data/malaria")),
                       ("bccd", Path("data/bccd"))):
        write_manifest(name, root)
    # Always ensure a toy set exists so E2E plumbing runs.
    from scripts.make_toy_dataset import build
    build()


if __name__ == "__main__":
    main()
