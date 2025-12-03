# Ensemble OCR wrapper using PaddleOCR, EasyOCR, and Tesseract (local)
from paddleocr import PaddleOCR
import easyocr
import pytesseract
import numpy as np

paddle = PaddleOCR(use_angle_cls=True, lang='en') 
reader = easyocr.Reader(['en'], gpu=False)  

def run_paddle(image):
    # image can be numpy array (BGR or gray) as returned by OpenCV
    res = paddle.ocr(image, cls=True)
    ocr_lines = []
    for line in res:
        box, (text, score) = line
        ocr_lines.append({'text': text, 'score': float(score), 'box': box, 'engine': 'paddle'})
    return ocr_lines

def run_easyocr(image):
    res = reader.readtext(image)
    ocr_lines = []
    for bbox, text, score in res:
        ocr_lines.append({'text': text, 'score': float(score), 'box': bbox, 'engine': 'easyocr'})
    return ocr_lines

def run_tesseract(image):
    config = '--psm 6'  # assume a single uniform block of text; we will try other psm if needed
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config, lang='eng')
    n = len(data['level'])
    lines = []
    for i in range(n):
        text = data['text'][i].strip()
        if text:
            score = float(data.get('conf', [0]*n)[i]) if data.get('conf') else 0
            box = [data['left'][i], data['top'][i], data['width'][i], data['height'][i]]
            lines.append({'text': text, 'score': score/100.0 if score!='' else 0.0, 'box': box, 'engine': 'tesseract'})
    return lines

def run_all_engines(img):
    results = []
    try:
        results += run_paddle(img)
    except Exception as e:
        print("PaddleOCR error:", e)
    try:
        results += run_easyocr(img)
    except Exception as e:
        print("EasyOCR error:", e)
    try:
        results += run_tesseract(img)
    except Exception as e:
        print("Tesseract error:", e)
    return results
