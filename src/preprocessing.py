import cv2
import numpy as np

def load_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def denoise(img):
    return cv2.fastNlMeansDenoising(img, h=10)

def clahe_equalize(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def adaptive_thresh(gray):
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)

def deskew(img_gray):
    coords = np.column_stack(np.where(img_gray < 255))
    if coords.shape[0] < 10:
        return img_gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img_gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_for_ocr(path):
    img = load_image(path)
    gray = to_grayscale(img)
    den = denoise(gray)
    clahe = clahe_equalize(den)
    desk = deskew(clahe)
    th = adaptive_thresh(desk)
    
    # return both original-color and processed binary for display / OCR experiments
    return img, desk, th
