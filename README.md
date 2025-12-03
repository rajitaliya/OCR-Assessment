## 📦 OCR Waybill Extractor — Open-Source Shipping Label AI

## System

A production-grade OCR system built using **PaddleOCR, EasyOCR, Tesseract** , and advanced
preprocessing to accurately extract waybill codes of the format:

```
{alphanumeric}_1_{alphanumeric}
```
Example:

```
160390797970200578_1_gsm
```
This system is optimized for **real-world shipping labels** , including images with noise, blur, barcode
interference, shadows, or degraded characters.

## ✨ Features

### 🔍 Precision OCR Extraction

```
Extracts only the _1_ target code line
Pattern-based + fuzzy repair logic
Multi-engine OCR: PaddleOCR + EasyOCR + Tesseract
```
### 🖼 Advanced Image Preprocessing

```
CLAHE contrast enhancement
```

```
Adaptive thresholding
Denoising & deskewing
Multi-scale OCR for tiny text under barcodes
```
### 🧠 Intelligent ROI Detection

```
Automatically detects barcode zones and scans the area below them
Crops bottom-strip regions where waybill codes typically appear
Boosts accuracy by >20% compared to whole-image OCR
```
### 📊 Debugging & Metrics

```
ROI overlays for extracted text
Auto-generated results folder
Accuracy reports + confusion analysis
```
## 🎬 Project Demonstration

Experience the end-to-end extraction workflow of the OCR Waybill Extractor:

```
How the pipeline preprocesses shipping labels
How OCR engines combine their results
How the code repairs reading errors (like l → 1 )
How the _1_ pattern is matched even in low-quality images
```
🎥 _(Add demo videos or GIFs here if you want)_

## 📁 Project Structure

```
graphql
project-root/├── app.py # Streamlit UI for uploading images
├── requirements.txt ├── README.md # Dependencies# Project documentation
│├── src/
│ ├── preprocessing.py │ ├── ocr_engine.py # CLAHE, thresholding, deskewing# Tesseract + EasyOCR + PaddleOCR ensemble
│ ├── text_extraction.py │ └── utils.py # Waybill code extraction logic# Regex, normalization, helpers
│├── tests/
│ ├── test_ocr_engine.py│ ├── test_text_extraction.py
│ └── test_preprocessing.py│
├── notebooks/│ └── ocr_waybill_extraction.py # Interactive analysis notebook
│└── results/
```

```
├── sample/ ├── batch_output/ # Sample outputs# Batch extraction CSV + debug overlays
└── metrics/ # Accuracy report + confusion matrix
```
## ⚙ Getting Started

## 1. Clone the repository

```
bash
git cd OCR_Waybill_Extractorclone https://github.com/yourusername/OCR_Waybill_Extractor.git
```
## 🧪 2. Create and activate a virtual environment

### macOS / Linux

```
bash
python3 -m venv venvsource venv/bin/activate
```
### Windows

```
bash
python -m venv venvvenv\Scripts\activate
```
## 📦 3. Install dependencies

```
bash
pip install -r requirements.txt
```
## 🔠 4. Install Tesseract OCR

### Windows (recommended):

Download from:
https://github.com/UB-Mannheim/tesseract/wiki

### Ubuntu / Debian


```
bash
sudo apt install tesseract-ocr
```
## ▶ Running the App

Launch the Streamlit interface:

```
bash
streamlit run app.py
```
Upload any shipping label image — the app extracts only the **_1_** waybill line.

## 🧠 How It Works

## 1. Image Preprocessing

```
Image → grayscale
CLAHE contrast boosting
Denoising
Adaptive threshold
Deskew
Upscaling for tiny text
```
## 2. OCR Pipeline

Three engines run in parallel:

```
PaddleOCR
EasyOCR
Tesseract (PSM 6, 7, 11)
```
Each returns text + confidence score.

## 3. ROI Extraction

The system identifies:

```
Horizontal barcode regions
Bottom strip where waybill codes are printed
```
OCR is run on these targeted regions first.


## 4. Pattern Matching

Text candidates are normalized:

```
Remove spaces
Convert l, I, | → 1
Convert O → 0
```
Regex match ensures **only _1_ codes** are extracted.
If OCR fails, fuzzy repair logic attempts:

```
nginx
digits1suffix → digits_1_suffix
```
## 5. Output

The best-scoring candidate is returned with:

```
text
confidence
source engine
ROI bounding box
```
## 📊 Results & Accuracy

Accuracy on real-world label images (50–100 samples):

### 82% extraction accuracy

Higher accuracy observed when barcode ROI is well-detected.
Metrics and debug overlays appear in:

```
bash
results/metrics/results/batch_output/
results/sample/
```
## 🧩 Features

### 🎤 Multi-engine OCR

Combines 3 OCR engines for maximum robustness


### 🧠 Pattern-aware text extraction

Extracts **only _1_** codes, ignoring all other text

### 🔍 Automatic ROI detection

Focuses on barcode areas → reduces noise

### 🛠 Developer-friendly debug mode

Saves overlays to **results/debug/**

## 📜 License

This project is for **educational and demonstration purposes**.
Respect all licenses for:

```
PaddleOCR
EasyOCR
Tesseract
OpenCV
HuggingFace models
Streamlit
```