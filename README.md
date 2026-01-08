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
pip install -r requirements.txt
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

