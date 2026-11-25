# Module 2 — Re-ID Embeddings + Face Detection + Search Engine

Generates 512-dimensional OSNet person re-identification embeddings, checks face visibility with MTCNN, and provides the `search()` engine used by Module 4.

## Setup

Install dependencies:

```bash
pip install torch torchvision
pip install torchreid
pip install facenet-pytorch
pip install numpy scipy pillow
```

If `torchreid` fails to install directly:

```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

## Input & Output

- **Inputs (from Module 1):**
  - `output/detections.json`
  - `output/crops/*.jpg`
- **Outputs:**
  - `output/embeddings.npy` — NumPy array of shape `(N, 512)`
  - `output/processed_detections.json` — `detections.json` extended with `embedding_index`, `face_visible`, `face_quality`, and `face_confidence`

## How to Run

### 1. Generate Embeddings & Face Visibility Metadata

```bash
cd module2_reid
python embedder.py --detections ../output/detections.json
```

### 2. Test Search Engine

Using a JSON query:

```bash
python search.py --query_json '{"upper_color":"blue","camera_id":"cam1"}'
```

Or using CLI flags:

```bash
python search.py --camera_id cam1 --time_from "20:00" --time_to "21:00"
```
