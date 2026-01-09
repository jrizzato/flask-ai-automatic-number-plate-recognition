# ANPR with Flask and Deep Learning

Simple Automatic Number Plate Recognition (ANPR) project that:

- Detects the license plate in an image using a CNN model (InceptionV3 as a feature extractor + dense head for bounding box regression).
- Extracts the plate text with OCR (EasyOCR).
- Serves a Flask web UI to upload images and view results.

## Requirements

- Python 3.9–3.11 (Windows recommended for this repo)
- Visual C++ Redistributable (on Windows, if TensorFlow requires it):
	https://aka.ms/vs/17/release/vc_redist.x64.exe
- Internet connection to download InceptionV3 weights on first run.

## Quick Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r deps/requirements.txt
```

## Expected Data

- Place your images (.jpg/.png) and PASCAL VOC annotations (.xml) in `data/images/`.
- The repo already includes many `.xml` files; ensure you have the matching images.

## Training Workflow

1) Build labels CSV from XML

```powershell
python modules/01-xml-to-csv.py
```

2) Prepare image paths and quick visual check

```powershell
python modules/02-object-detection.py
```

3) Preprocess data (normalization and split)

```powershell
python modules/03-data-preprocesing.py
```

4) Build the model (frozen InceptionV3 + dense head)

```powershell
python modules/04-deep-learning-model.py
```

5) Train the model and log to TensorBoard

```powershell
python modules/05-model-training.py
```

Open TensorBoard (optional):

```powershell
tensorboard --logdir object_detection
```

## Quick Tests (prediction and OCR)

- Predict a bounding box on a validation image:

```powershell
python modules/06-make-prediction.py
```

- Detection + OCR with EasyOCR and ROI visualization:

```powershell
python modules/07-OCR.py
```

## Run the Web App (Flask)

```powershell
python server.py
```

Then open http://localhost:5000 and upload an image. The app will save:

- The image with bounding box to `static/predict/`.
- The plate ROI to `static/roi/`.

## Docker & Docker Compose

### Using Docker Compose (recommended)

Build and run the web app with TensorBoard in one command:

```bash
docker compose up --build
```

- **Web app**: http://localhost:5000
- **TensorBoard**: http://localhost:6006 (for monitoring training logs)

To run the training pipeline in a container (one-time job):

```bash
docker compose --profile train run train
```

### Using Docker alone (without Compose)

Build the image:

```bash
docker build -t anpr-flask .
```

Run the Flask web app:

```bash
docker run --rm -p 5000:5000 \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/model:/app/model \
  -v ${PWD}/static:/app/static \
  anpr-flask
```

Run a one-off training:

```bash
docker run --rm \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/model:/app/model \
  -v ${PWD}/object_detection:/app/object_detection \
  anpr-flask python modules/05-model-training.py
```

### Important Notes for Docker

- **Volumes**: The examples above mount local directories (`./data`, `./model`, `./static`, `./object_detection`) into the container so that trained models, data, and logs persist on your host machine.
- **Docker Compose**: See `docker-compose.yml` for detailed configuration of each service (web, tensorboard, train) with all comments.
- **Windows**: Replace `${PWD}` with `%cd%` or use PowerShell with `$PWD` if necessary.

## Project Structure (overview)

```
server.py
modules/
	01-xml-to-csv.py
	02-object-detection.py
	03-data-preprocesing.py
	04-deep-learning-model.py
	05-model-training.py
	06-make-prediction.py
	07-OCR.py
data/
	images/                # images + XML (PASCAL VOC)
	labels.csv             # generated
	*.joblib               # intermediate artifacts
model/
	object_detection_model.joblib
static/
	uploaded/ predict/ roi/
templates/
	index.html
```

## Notes & Troubleshooting

- If TensorFlow import fails on Windows, install the Visual C++ redistributable linked above and try again.
- EasyOCR will download OCR models on first use (requires Internet). Installation may take longer due to PyTorch.
- If `cv2.imshow` windows do not appear (step 2), it is not blocking for the pipeline; they are only for visual inspection.

## Acknowledgements

This README was drafted with assistance from GitHub Copilot (AI).

---

License: Demo use.

