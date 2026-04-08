"""Training curriculum helpers (e.g. DINO semantic overlap profiles)."""

from erayzer_core.curriculum.dino_semantic_overlap import (
    build_semantic_overlap_profile,
    embed_image_paths_dinov3,
    load_dinov3_encoder,
    manifest_to_image_paths,
)

__all__ = [
    "build_semantic_overlap_profile",
    "embed_image_paths_dinov3",
    "load_dinov3_encoder",
    "manifest_to_image_paths",
]
