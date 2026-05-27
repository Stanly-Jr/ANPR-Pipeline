# 🚗 Number Plate Recognition (OCR) App

A custom end-to-end Machine Learning web application that detects, extracts, and recognizes alphanumeric characters from vehicle number plates. 

This project features a custom Deep Convolutional Neural Network (CNN) built from scratch using **PyTorch**, custom character segmentation using **OpenCV** Connected Component Analysis (CCA), and a clean, interactive web interface built with **Streamlit**.

## ✨ Features
* **Custom CNN Architecture:** A deep PyTorch model trained to classify 36 alphanumeric characters (0-9, A-Z).
* **Robust Character Extraction:** Uses OpenCV to convert images to grayscale, apply Otsu's thresholding, and perform Connected Component Analysis to isolate individual characters based on area, aspect ratio, and density heuristics.
* **Smart Filtering:** Automatically detects and filters out regional country codes (e.g., "IND") from the final prediction.
* **Interactive Web Interface:** A user-friendly Streamlit app where users can upload an image and instantly see both the final predicted text and a visual breakdown of the individual cropped characters.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, TorchVision
* **Computer Vision:** OpenCV (cv2)
* **Data Manipulation & Visualization:** NumPy, Matplotlib, Pillow
* **Web Framework:** Streamlit

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed. It is highly recommended to use a virtual environment.

### 1. Project Structure
Ensure your project directory looks like this:
```text
your-folder/
├── ocr.py                     # The Streamlit application script
├── ocr_model_deep_v1.pth      # Your trained PyTorch model weights
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

### 2. Running the File
```bash
streamlit run ocr.py
```
