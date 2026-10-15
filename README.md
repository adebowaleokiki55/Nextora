# Nextora — Experimental AI-Powered Malaria Blood Smear Analysis

Nextora is a machine learning application that classifies blood cell images into parasitized and uninfected classes and provides experimental candidate-region analysis for whole blood-smear images.

The project combines a PyTorch image classification model with a FastAPI backend and a React frontend to make malaria-related image analysis accessible through a web interface.

**Live Demo:** https://nextora-zeta.vercel.app/

> **Important:** Nextora is an experimental research and educational prototype, not a clinically validated diagnostic tool. Its predictions must not be used as a substitute for professional medical assessment or laboratory confirmation.

## Features

- **Single-cell classification:** Upload a blood cell image and receive a model prediction.
- **Whole-smear analysis:** Experimentally identifies candidate regions in a larger blood-smear image and classifies image crops.
- **Visual analysis:** Displays an annotated image showing candidate regions detected by the analysis pipeline.
- **Prediction confidence:** Presents the model's confidence scores where available.
- **Interactive web interface:** Upload images and inspect analysis results through a React-based frontend.
- **REST API:** Uses FastAPI to expose prediction and smear-analysis endpoints.
- **Deep learning:** Uses a custom PyTorch model, `PlasmoSENet`, for two-class image classification.

## How It Works

### 1. Single-cell classification

1. The user uploads a blood cell image.
2. The image is resized to 224 × 224 pixels.
3. The image is converted to a tensor and normalized using ImageNet mean and standard deviation values.
4. The trained `PlasmoSENet` model generates a prediction.
5. The application returns the predicted class and available confidence scores.

### 2. Whole-smear analysis

1. The user uploads a larger blood-smear image.
2. The backend uses OpenCV-based image processing and contour heuristics to identify candidate regions.
3. Candidate regions are cropped from the original image.
4. The existing classification model predicts a class for each crop.
5. The backend returns the candidate-region results, counts, and an annotated image.

**Important limitation:** Candidate regions are not guaranteed to correspond to individual cells. The whole-smear pipeline is experimental, and classifying crops from full-smear images may produce different results from classifying isolated cell images.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| Image processing | OpenCV |
| Machine learning utilities | scikit-learn |
| Backend API | FastAPI |
| Frontend | React, Vite |
| Frontend styling | CSS |
| Frontend hosting | Vercel |
| Backend hosting | Render |

## Architecture

```text
                 User
                  |
                  v
       React + Vite Frontend
                  |
                  | HTTP requests
                  v
           FastAPI Backend
             /         \
            v           v
     Single-cell    Whole-smear
     prediction      analysis
            |           |
            v           v
        PlasmoSENet + OpenCV
                  |
                  v
       Prediction and image results
                  |
                  v
          Frontend display
```

## Model Information

The application uses `PlasmoSENet`, a PyTorch image classification model configured for two classes.

| Class ID | Class label |
|---|---|
| 0 | Parasitized (Infected) |
| 1 | Uninfected |

**Image preprocessing**

- Resize: 224 × 224 pixels
- Convert image to tensor
- Normalize using ImageNet statistics

Mean:

```python
[0.485, 0.456, 0.406]
```

Standard deviation:

```python
[0.229, 0.224, 0.225]
```

Model performance metrics are not included here because independently verified evaluation results have not been provided. Accuracy, precision, recall, F1-score, and confusion-matrix results should be added after evaluation on an appropriately separated test dataset.

## Project Structure

```text
Malaria_Detection/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── smear_analysis.py
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       └── index.css
├── model.pth
├── plasmosenet.py
├── run_prediction.py
├── cell_image.png
└── README.md
```

The structure above highlights the main project files. Other configuration files, dependencies, and frontend assets may also be present.

## Running Locally

### Prerequisites

- Python 3.11 recommended
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/adebowaleokiki55/Nextora.git
cd Nextora
```

### 2. Set up the Python backend

Create and activate a virtual environment from the project root.

**Windows PowerShell**

```powershell
python -m venv backend/.venv
.\backend\.venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```powershell
pip install -r backend/requirements.txt
```

Ensure `model.pth` and `plasmosenet.py` are available at the paths expected by the backend.

### 3. Configure environment variables

Configure the frontend API URL to point to your local FastAPI server.

Create a `.env` file inside `frontend/`:

```env
VITE_API_URL=http://127.0.0.1:8000
```

For production, set `VITE_API_URL` to the deployed backend URL in your hosting provider's environment-variable settings.

### 4. Start the backend

From the project root, run:

```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

If your `main.py` uses imports or paths that require running from inside `backend/`, adjust the launch command accordingly.

The local API should be available at:

- API: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs

### 5. Start the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite in your terminal.

## API Endpoints

The FastAPI backend exposes the following endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Check backend health |
| POST | `/api/predict` | Classify a single-cell image |
| POST | `/api/analyze-smear` | Perform experimental whole-smear candidate-region analysis |

Open `/docs` on the running backend to inspect the request schemas, required image-upload fields, and response formats.

## Deployment

The current application is deployed using:

- **Frontend:** Vercel — https://nextora-zeta.vercel.app/
- **Backend:** Render — https://nextora-npmm.onrender.com

The frontend sends API requests to the backend. The production API URL and CORS configuration must match the deployed frontend domain.

## Validation and Limitations

Nextora is a prototype, and several limitations must be considered:

- A predicted class is not a confirmed medical diagnosis.
- Model confidence is not equivalent to clinical certainty.
- Whole-smear candidate detection uses image-processing heuristics and may miss cells or identify non-cell regions.
- A classifier trained on isolated cell images may not generalize reliably to crops extracted from whole-smear images.
- Clinical use would require rigorous validation, independent testing, expert-annotated data, and assessment across different imaging conditions and patient populations.

Future evaluation should include patient-independent train/test splits to reduce data leakage, per-class precision and recall, F1-score, a confusion matrix, and external validation where appropriate.

## Future Improvements

- Evaluate generalization on unseen patient samples.
- Improve candidate-cell detection and localization.
- Compare predictions against expert-annotated microscopy images.
- Add model evaluation reports and error analysis.
- Improve inference performance and image-upload validation.
- Investigate whether the whole-smear pipeline can support reliable, validated quantitative measurements.

## Disclaimer
Nextora is intended for research, learning, and demonstration purposes only. It has not been established as a clinically validated diagnostic system. Do not use its output to diagnose, treat, or rule out malaria. Medical decisions must be made with qualified healthcare professionals and appropriate laboratory testing.


**Okiki** — Data Science student interested in machine learning, artificial intelligence, and practical applications of AI.

GitHub: https://github.com/adebowaleokiki55

