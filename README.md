# Intelligent CCTV Forensics System

Natural Language Search and AI-Powered Video Investigation Platform.

## Overview

The Intelligent CCTV Forensics System makes recorded CCTV footage searchable using plain English queries (e.g., `"person in blue jacket near main gate between 8 and 9 PM"`). It indexes video footage in a compact SQLite + NumPy representation, ranks matches with Re-ID embeddings, reconstructs cross-camera movement timelines, flags anomalous behaviour, highlights clips with visible faces, and exports PDF incident reports.

## Team & Module Structure

```text
cctv_forensics/
├── config.py                  # Shared configuration (paths, cameras, thresholds)
├── module1_detection/         # Video ingestion, OCR timestamps, YOLOv8 person crops
├── module2_reid/              # OSNet Re-ID embeddings, MTCNN face check, search()
├── module3_attributes/        # ResNet-50 clothing attribute classifier & spaCy NLP parser
└── module4_backend/           # SQLite DB, Isolation Forest anomalies, trajectory, PDF, Flask app
    └── templates/             # HTML templates for the Flask web interface
```

| Module | Responsibility |
| --- | --- |
| `module1_detection` | Frame extraction, OCR/fallback timestamps, YOLOv8 person detection |
| `module2_reid` | 512-dim OSNet embeddings, MTCNN face visibility, cosine similarity search |
| `module3_attributes` | PA-100K ResNet-50 attribute classifier & spaCy natural language query parser |
| `module4_backend` | SQLite index, Isolation Forest anomaly detection, trajectory, ReportLab PDF, Flask UI |

## Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install ultralytics torch torchvision torchreid facenet-pytorch opencv-python pytesseract pillow spacy scikit-learn flask reportlab numpy scipy requests tqdm
   python -m spacy download en_core_web_sm
   ```
3. Verify the shared configuration:
   ```bash
   python config.py
   ```

## Running the System

1. **Module 1 (Detection):**
   ```bash
   cd module1_detection
   python detector.py --video /path/to/video.mp4 --camera cam1
   ```
2. **Module 2 (Re-ID Embeddings & Search):**
   ```bash
   cd module2_reid
   python embedder.py --detections ../output/detections.json
   ```
3. **Module 4 (Import & Web App):**
   ```bash
   cd module4_backend
   python import_data.py
   python app.py
   ```
