# =============================================================================
# config.py  -  SHARED BY ALL 4 MODULES
# Intelligent CCTV Forensics System
#
# RULES
#   1. Put this file in the ROOT of the repo (same level as module1_detection/).
#   2. Every module imports from here. Never hardcode a path or number.
#   3. Only the team leader changes this file. Tell everyone when it changes.
#   4. Run `python config.py` to check that your setup is correct.
#
# HOW TO IMPORT (from inside any module folder):
#       import sys, os
#       sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#       import config
# =============================================================================

import os

# -----------------------------------------------------------------------------
# PATHS
# BASE_DIR is the repo root, worked out automatically from where this file is.
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_FOLDER      = os.path.join(BASE_DIR, "output")
CROPS_FOLDER       = os.path.join(OUTPUT_FOLDER, "crops")
FRAMES_FOLDER      = os.path.join(OUTPUT_FOLDER, "frames")
MODELS_FOLDER      = os.path.join(BASE_DIR, "models")
REPORTS_FOLDER     = os.path.join(BASE_DIR, "reports")
UPLOADS_FOLDER     = os.path.join(BASE_DIR, "uploads")

DATABASE_PATH      = os.path.join(BASE_DIR, "forensics.db")
EMBEDDINGS_PATH    = os.path.join(OUTPUT_FOLDER, "embeddings.npy")
DETECTIONS_JSON    = os.path.join(OUTPUT_FOLDER, "detections.json")            # Module 1 output
PROCESSED_JSON     = os.path.join(OUTPUT_FOLDER, "processed_detections.json")  # Module 2 output

OSNET_FINETUNED_PATH = os.path.join(MODELS_FOLDER, "osnet_finetuned.pth")
ATTRIBUTE_MODEL_PATH = os.path.join(MODELS_FOLDER, "attribute_model.pth")
ANOMALY_MODEL_PATH   = os.path.join(MODELS_FOLDER, "anomaly_model.pkl")

# Create the folders automatically so nobody gets "folder not found" errors
for _folder in (OUTPUT_FOLDER, CROPS_FOLDER, FRAMES_FOLDER, MODELS_FOLDER,
                REPORTS_FOLDER, UPLOADS_FOLDER):
    os.makedirs(_folder, exist_ok=True)


# -----------------------------------------------------------------------------
# PATH HELPERS  (solves the relative vs absolute path problem)
#
# RULE: inside the JSON files, crop_path is stored RELATIVE to the repo root,
#       e.g.  "output/crops/cam1_frame000100_person0.jpg"
#       This keeps the JSON portable between everyone's laptops.
#
#       When you need to actually OPEN the image, wrap it:
#           img = cv2.imread(config.to_abs(detection["crop_path"]))
#       This works no matter which folder you run your script from.
# -----------------------------------------------------------------------------
def to_abs(path):
    """Turn a repo-relative path into a full path that works from anywhere."""
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def to_rel(path):
    """Turn a full path into a repo-relative path (use this before saving to JSON)."""
    return os.path.relpath(path, BASE_DIR).replace("\\", "/")


# -----------------------------------------------------------------------------
# CAMERA SETUP
# Change these to match your real campus cameras before the demo.
# -----------------------------------------------------------------------------
CAMERA_LOCATIONS = {
    "cam1": "Main Gate",
    "cam2": "Parking Lot",
    "cam3": "Canteen",
    "cam4": "Library Entrance",
}

# Pairs of cameras that are far apart. If the same person shows up in both
# within IMPOSSIBLE_TRANSITION_SECONDS, flag it as a possible false match.
FAR_APART_CAMERAS = [
    ("cam1", "cam4"),
    ("cam2", "cam3"),
]
IMPOSSIBLE_TRANSITION_SECONDS = 120


# -----------------------------------------------------------------------------
# MODULE 1  -  DETECTION SETTINGS
# -----------------------------------------------------------------------------
FRAME_INTERVAL_SECONDS   = 5      # take one frame every N seconds
MIN_DETECTION_CONFIDENCE = 0.5    # YOLOv8 minimum confidence to keep a person
PERSON_CROP_WIDTH        = 128    # every saved crop is resized to this width
PERSON_CROP_HEIGHT       = 256    # ... and this height
YOLO_MODEL_NAME          = "yolov8n.pt"
TIMESTAMP_CROP_HEIGHT    = 60     # OCR looks at a strip this tall in the corners
TIMESTAMP_CROP_WIDTH     = 350    # ... and this wide


# -----------------------------------------------------------------------------
# MODULE 2  -  RE-ID, FACE DETECTION, SEARCH
# -----------------------------------------------------------------------------
EMBEDDING_DIM             = 512
TOP_K_RESULTS             = 5
REID_SIMILARITY_THRESHOLD = 0.7    # min cosine similarity to count as same person
IMAGENET_MEAN             = [0.485, 0.456, 0.406]
IMAGENET_STD              = [0.229, 0.224, 0.225]

MIN_FACE_CONFIDENCE       = 0.90   # MTCNN must be at least this sure
MIN_FACE_SIZE_PIXELS      = 10     # MTCNN min_face_size (small, CCTV faces are tiny)
FACE_HIGH_QUALITY_RATIO   = 0.15   # face area > 15% of crop  -> "High"
FACE_MED_QUALITY_RATIO    = 0.05   # face area 5% - 15%       -> "Medium", below -> "Low"

REID_LEARNING_RATE = 0.0001
REID_EPOCHS        = 25
REID_BATCH_SIZE    = 32


# -----------------------------------------------------------------------------
# MODULE 3  -  ATTRIBUTES AND NLP
# The ORDER of these lists matters: the position in the list is the class number
# the model is trained on. Do not reorder after training starts.
# -----------------------------------------------------------------------------
UPPER_COLORS = ["red", "blue", "green", "black", "white", "grey", "yellow", "other"]
UPPER_TYPES  = ["tshirt", "shirt", "jacket", "hoodie", "other"]
LOWER_COLORS = ["red", "blue", "green", "black", "white", "grey", "yellow", "other"]
LOWER_TYPES  = ["jeans", "trousers", "shorts", "other"]

ATTR_LEARNING_RATE = 0.0001
ATTR_EPOCHS        = 20
ATTR_BATCH_SIZE    = 32

# Words the NLP parser looks for in a query  ->  value it returns
COLOR_KEYWORDS = {
    "red": "red", "blue": "blue", "green": "green", "black": "black",
    "white": "white", "grey": "grey", "gray": "grey", "yellow": "yellow",
}

UPPER_CLOTHING_KEYWORDS = {
    "jacket": "jacket", "coat": "jacket", "shirt": "shirt",
    "tshirt": "tshirt", "t-shirt": "tshirt", "hoodie": "hoodie",
}

LOWER_CLOTHING_KEYWORDS = {
    "jeans": "jeans", "trousers": "trousers", "pants": "trousers",
    "shorts": "shorts",
}

# Words in a query that mean "only show clips with a visible face"
FACE_KEYWORDS = ["face visible", "clear face", "with face", "face shown", "visible face"]

# Extra names people might type for a location  ->  camera id
LOCATION_ALIASES = {
    "main gate": "cam1", "gate": "cam1", "entrance": "cam1",
    "parking": "cam2", "parking lot": "cam2",
    "canteen": "cam3", "cafeteria": "cam3",
    "library": "cam4",
}


# -----------------------------------------------------------------------------
# MODULE 4  -  ANOMALY DETECTION
# -----------------------------------------------------------------------------
ANOMALY_CONTAMINATION   = 0.05    # expect about 5% of movements to be unusual
ANOMALY_SCORE_THRESHOLD = -0.1    # Isolation Forest score below this = flagged
LOITERING_SECONDS       = 1800    # staying 30+ min in one zone = loitering


# -----------------------------------------------------------------------------
# QUICK SELF-CHECK:  python config.py
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Repo root          :", BASE_DIR)
    print("Crops folder       :", CROPS_FOLDER)
    print("Detections JSON    :", DETECTIONS_JSON)
    print("Cameras            :", CAMERA_LOCATIONS)
    print("Crop size          :", PERSON_CROP_WIDTH, "x", PERSON_CROP_HEIGHT)
    print("to_abs example     :", to_abs("output/crops/test.jpg"))
    print("All folders created. config.py is working.")
