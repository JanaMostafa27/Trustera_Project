# Fake News Detection System

A production-ready deep learning system for detecting fake news and classifying news categories. The model is based on a CNN-LSTM architecture and supports both batch processing and real-time predictions through an API.

---

## Features

* Fake vs True news classification
* News category prediction (multi-task learning)
* CNN-LSTM architecture with attention
* FastAPI-based REST API
* Batch and real-time inference
* Evaluation with accuracy and F1-score

---

## Project Structure

```
fake-news-detection/
├── data/        # Dataset (not included in repo)
├── models/      # Trained models
├── src/         # Core ML code
├── api/         # API service
├── utils/       # Utilities
├── main.py      # CLI entry point
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset is not included due to size limitations.

Download it from:
https://www.kaggle.com/datasets/janamostafa27/preprocessed-news

After downloading, place it in:

```
data/
```

---

## ▶Usage

### Train the model

```bash
python main.py train --data data/dataset.csv --model models/model.h5
```

### Evaluate the model

```bash
python main.py evaluate --model models/model.h5 --data data/test.csv
```

### Predict

```bash
python main.py predict --model models/model.h5 --text "Your news text here"
```

### Run API

```bash
python main.py api --model models/model.h5
```

---

## API Example

**Request:**

```json
{
  "text": "Sample news article..."
}
```

**Response:**

```json
{
  "prediction": "TRUE",
  "confidence": 0.85,
  "category": "politics"
}
```

---

## Notes

* Dataset is excluded using `.gitignore`
* Designed for research and educational use
* Ensure proper preprocessing before training

---
