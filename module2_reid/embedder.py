"""
Module 2: Re-ID Embedding Generation and Face Visibility Detection.

Generates 512-dimensional OSNet person re-identification embeddings and
MTCNN face visibility metadata for every person crop produced by Module 1.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torchreid
import torchvision.transforms as T
from facenet_pytorch import MTCNN
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

_CROP_TRANSFORM = T.Compose(
    [
        T.Resize((config.PERSON_CROP_HEIGHT, config.PERSON_CROP_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ]
)

_MTCNN_INSTANCE = None


def _get_mtcnn():
    """Return a cached MTCNN detector instance so it is loaded only once."""
    global _MTCNN_INSTANCE
    if _MTCNN_INSTANCE is None:
        _MTCNN_INSTANCE = MTCNN(
            keep_all=False,
            min_face_size=config.MIN_FACE_SIZE_PIXELS,
        )
    return _MTCNN_INSTANCE


def load_osnet_model():
    """
    Load the OSNet Re-ID model (osnet_x1_0) with pre-trained weights in eval mode.
    If fine-tuned campus weights exist at config.OSNET_FINETUNED_PATH, load them.
    """
    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1000,
        pretrained=True,
    )
    if os.path.exists(config.OSNET_FINETUNED_PATH):
        state_dict = torch.load(config.OSNET_FINETUNED_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_crop(crop_path):
    """
    Load a person crop image, resize it to (256, 128), normalize with ImageNet
    statistics, and return a PyTorch tensor of shape (1, 3, 256, 128).
    """
    abs_path = config.to_abs(crop_path)
    image = Image.open(abs_path).convert("RGB")
    tensor = _CROP_TRANSFORM(image)
    return tensor.unsqueeze(0)


def generate_embedding(crop_path, model):
    """
    Pass a person crop through the loaded OSNet model and return an
    L2-normalized 512-dimensional numpy embedding vector.
    """
    tensor = preprocess_crop(crop_path)
    with torch.no_grad():
        features = model(tensor)
    features = features.squeeze().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    return features


def detect_face(crop_path, mtcnn=None):
    """
    Check whether a clearly visible face is present in a person crop using MTCNN.
    Does NOT perform face recognition; only returns visibility, quality, and confidence.
    """
    detector = mtcnn if mtcnn is not None else _get_mtcnn()
    abs_path = config.to_abs(crop_path)
    image = Image.open(abs_path).convert("RGB")
    boxes, probs = detector.detect(image)

    if boxes is None or probs is None or probs[0] is None or probs[0] < config.MIN_FACE_CONFIDENCE:
        return {
            "face_visible": False,
            "face_quality": None,
            "face_confidence": 0.0,
        }

    width, height = image.size
    face_box = boxes[0]
    face_width = max(0.0, float(face_box[2] - face_box[0]))
    face_height = max(0.0, float(face_box[3] - face_box[1]))
    face_area = face_width * face_height
    total_area = width * height
    ratio = face_area / total_area if total_area > 0 else 0.0

    if ratio > config.FACE_HIGH_QUALITY_RATIO:
        quality = "High"
    elif ratio > config.FACE_MED_QUALITY_RATIO:
        quality = "Medium"
    else:
        quality = "Low"

    return {
        "face_visible": True,
        "face_quality": quality,
        "face_confidence": float(probs[0]),
    }


def process_all_crops(detections_json_path=config.DETECTIONS_JSON):
    """
    Run OSNet embedding generation and MTCNN face detection on every crop listed
    in detections.json, saving embeddings.npy and processed_detections.json.
    Returns the total number of processed crops.
    """
    abs_detections_path = config.to_abs(detections_json_path)
    with open(abs_detections_path, "r", encoding="utf-8") as f:
        detections = json.load(f)

    if not detections:
        empty_embeddings = np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)
        np.save(config.EMBEDDINGS_PATH, empty_embeddings)
        with open(config.PROCESSED_JSON, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        print("Processed 0 crops. Embeddings saved.")
        return 0

    # Load models once before the loop for fast CPU inference
    model = load_osnet_model()
    mtcnn = _get_mtcnn()

    embeddings_list = []
    processed_detections = []

    for index, detection in enumerate(detections):
        crop_path = detection["crop_path"]
        embedding = generate_embedding(crop_path, model)
        face_info = detect_face(crop_path, mtcnn=mtcnn)

        embeddings_list.append(embedding)

        updated = dict(detection)
        updated["embedding_index"] = index
        updated["face_visible"] = face_info["face_visible"]
        updated["face_quality"] = face_info["face_quality"]
        updated["face_confidence"] = face_info["face_confidence"]
        processed_detections.append(updated)

    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    np.save(config.EMBEDDINGS_PATH, embeddings_array)

    with open(config.PROCESSED_JSON, "w", encoding="utf-8") as f:
        json.dump(processed_detections, f, indent=2)

    print(f"Processed {len(processed_detections)} crops. Embeddings saved.")
    return len(processed_detections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Re-ID embeddings and face visibility flags for person crops."
    )
    parser.add_argument(
        "--detections",
        default=config.DETECTIONS_JSON,
        help="Path to Module 1 detections.json",
    )
    args = parser.parse_args()
    count = process_all_crops(args.detections)
    print(f"Done. {count} crops processed.")
