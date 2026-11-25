"""Module 2: Re-ID Embeddings, Face Detection, and Search Engine."""

from .embedder import (
    detect_face,
    generate_embedding,
    load_osnet_model,
    preprocess_crop,
    process_all_crops,
)
from .search import cosine_similarity, search

__all__ = [
    "load_osnet_model",
    "preprocess_crop",
    "generate_embedding",
    "detect_face",
    "process_all_crops",
    "cosine_similarity",
    "search",
]
