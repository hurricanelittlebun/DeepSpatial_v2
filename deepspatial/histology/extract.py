"""CLI: python -m deepspatial.histology.extract MANIFEST.json FEATURES.h5."""

import argparse
import json
import logging
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Offline UNI2 grids from already-registered OpenSlide-readable images"
    )
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Approved local UNI2-h pytorch_model.bin; otherwise gated HF load",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    from .feature_store import FeatureStore
    from .uni2 import UNI2Encoder
    from .preprocessing import precompute_section
    import openslide

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(str(output) + ".log")],
    )
    entries = json.loads(Path(args.manifest).read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must be a nonempty JSON list")
    if any(e.get("registered") is not True for e in entries):
        raise ValueError(
            "Each entry must explicitly declare registered=true; raw SDPC is not registered WSI"
        )
    store = FeatureStore(output, mode="a")
    encoder = None
    for entry in entries:
        meta = dict(entry)
        meta.pop("registered")
        source = Path(meta.pop("image_path")).resolve()
        sid = str(meta.pop("section_id"))
        mask_path = meta.pop("valid_mask_path", None)
        provenance = dict(
            source=str(source),
            source_bytes=source.stat().st_size,
            source_mtime_ns=source.stat().st_mtime_ns,
            encoder="UNI2-h",
            checkpoint=args.checkpoint,
            pipeline="physical-fov-v1",
            manifest_entry=entry,
        )
        if args.checkpoint:
            ckpt = Path(args.checkpoint)
            provenance.update(
                checkpoint_bytes=ckpt.stat().st_size,
                checkpoint_mtime_ns=ckpt.stat().st_mtime_ns,
            )
        if mask_path:
            mask = Path(mask_path)
            provenance.update(
                mask_bytes=mask.stat().st_size, mask_mtime_ns=mask.stat().st_mtime_ns
            )
        if sid in store.metadata:
            if store.metadata[sid]["provenance"] != provenance:
                raise ValueError(
                    f"{sid}: preprocessing inputs changed; write a new feature store"
                )
            logging.info("Reusing completed section %s", sid)
            continue
        if encoder is None:
            encoder = UNI2Encoder(args.checkpoint, device=args.device)
        reader = openslide.OpenSlide(str(source))
        try:
            precompute_section(
                reader,
                encoder,
                store,
                sid,
                valid_mask=(
                    np.load(mask_path, allow_pickle=False) if mask_path else None
                ),
                batch_size=args.batch_size,
                provenance=provenance,
                **meta,
            )
        finally:
            reader.close()


if __name__ == "__main__":
    main()
