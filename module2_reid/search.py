"""
Module 2: Re-ID Search Engine.

Filters indexed person detections by structured query parameters and ranks
matches using cosine similarity over 512-dimensional OSNet embeddings.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial.distance import cosine

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two embedding vectors of shape (512,).
    Returns a float in [0.0, 1.0] where 1.0 means identical.
    """
    v1 = np.asarray(vec1, dtype=np.float32)
    v2 = np.asarray(vec2, dtype=np.float32)
    if np.linalg.norm(v1) == 0.0 or np.linalg.norm(v2) == 0.0:
        return 0.0
    return float(1.0 - cosine(v1, v2))


def _extract_time_str(timestamp):
    """Extract the HH:MM:SS portion from a 'YYYY-MM-DD HH:MM:SS' timestamp."""
    if not timestamp:
        return ""
    if " " in timestamp:
        return timestamp.split(" ", 1)[1]
    return timestamp


def _normalize_time_bound(time_str, is_upper=False):
    """Normalize HH:MM to HH:MM:00 (or HH:MM:59 for upper bound) for string comparison."""
    if len(time_str) == 5:
        return f"{time_str}:59" if is_upper else f"{time_str}:00"
    return time_str


def search(query_params, top_k=config.TOP_K_RESULTS):
    """
    Filter processed detections using structured query parameters from Module 3's
    parse_query(), compute cosine similarity against the query appearance vector,
    and return the top_k matching results sorted by confidence descending.
    """
    if not os.path.exists(config.PROCESSED_JSON) or not os.path.exists(config.EMBEDDINGS_PATH):
        return []

    with open(config.PROCESSED_JSON, "r", encoding="utf-8") as f:
        detections = json.load(f)

    if not detections:
        return []

    embeddings = np.load(config.EMBEDDINGS_PATH)
    if len(embeddings) == 0:
        return []

    params = query_params or {}
    time_from = params.get("time_from")
    time_to = params.get("time_to")
    camera_id = params.get("camera_id")
    face_visible = bool(params.get("face_visible", False))

    # Step 2: Hard filter by time window, camera, and face visibility
    filtered = []
    for det in detections:
        det_time = _extract_time_str(det.get("timestamp", ""))
        if time_from and det_time < _normalize_time_bound(time_from, is_upper=False):
            continue
        if time_to and det_time > _normalize_time_bound(time_to, is_upper=True):
            continue
        if camera_id and det.get("camera_id") != camera_id:
            continue
        if face_visible and not det.get("face_visible", False):
            continue
        filtered.append(det)

    # Step 3: Build query vector from mean of filtered embeddings (or all as fallback)
    if filtered:
        filtered_indices = [det["embedding_index"] for det in filtered]
        query_vec = np.mean(embeddings[filtered_indices], axis=0)
    else:
        query_vec = np.mean(embeddings, axis=0)
        return []

    # Step 4: Score each remaining detection and format result dicts
    scored_results = []
    for det in filtered:
        emb_idx = det["embedding_index"]
        similarity = cosine_similarity(query_vec, embeddings[emb_idx])
        confidence = round(max(0.0, similarity) * 100.0, 2)

        scored_results.append(
            {
                "detection_id": det.get("detection_id", det.get("id", emb_idx + 1)),
                "crop_path": det["crop_path"],
                "camera_id": det["camera_id"],
                "timestamp": det["timestamp"],
                "confidence": confidence,
                "face_visible": bool(det.get("face_visible", False)),
                "face_quality": det.get("face_quality"),
                "embedding_index": emb_idx,
            }
        )

    # Step 5: Sort by confidence descending and return top_k
    scored_results.sort(key=lambda item: item["confidence"], reverse=True)
    return scored_results[:top_k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search indexed CCTV detections using structured query parameters."
    )
    parser.add_argument(
        "--query_json",
        default=None,
        help="JSON string matching parse_query() output",
    )
    parser.add_argument("--query", default=None, help="Optional text description")
    parser.add_argument("--time_from", default=None, help="Start time HH:MM")
    parser.add_argument("--time_to", default=None, help="End time HH:MM")
    parser.add_argument("--camera_id", default=None, help="Camera ID e.g. cam1")
    parser.add_argument(
        "--face_visible",
        action="store_true",
        help="Only return detections with a visible face",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=config.TOP_K_RESULTS,
        help="Number of top results to return",
    )
    args = parser.parse_args()

    if args.query_json:
        query_dict = json.loads(args.query_json)
    else:
        query_dict = {
            "upper_color": None,
            "upper_type": None,
            "lower_color": None,
            "lower_type": None,
            "time_from": args.time_from,
            "time_to": args.time_to,
            "camera_id": args.camera_id,
            "face_visible": args.face_visible,
        }

    matches = search(query_dict, top_k=args.top_k)
    print(json.dumps(matches, indent=2))
