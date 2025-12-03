import cv2
import numpy as np
import pytesseract
from .preprocessing import preprocess_for_ocr
from .ocr_engine import run_all_engines
import re
from difflib import SequenceMatcher
import os

CODE_REGEX = re.compile(r'\b[A-Za-z0-9]+_1_[A-Za-z0-9_]+\b')

CONFUSION_MAP = str.maketrans({'I':'1', 'l':'1', '|':'1', '!':'1', 'O':'0', 'o':'0'})

def normalize(t: str) -> str:
    if not t:
        return ""
    t = t.strip()
    t = t.replace(" ", "")
    t = t.translate(CONFUSION_MAP)
    return t

def is_code(s: str) -> bool:
    return bool(CODE_REGEX.search(s))

def _tesseract_lines(img, psm=6, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"):
    """
    Return list of (text, conf) from pytesseract line-level detection.
    """
    config = f'--psm {psm} -c tessedit_char_whitelist={whitelist}'
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config, lang='eng')
    out = []
    n = len(data.get('text', []))
    for i in range(n):
        txt = str(data['text'][i]).strip()
        if not txt:
            continue
        try:
            conf = float(data['conf'][i])
        except:
            conf = 0.0
        out.append((txt, conf/100.0 if conf >= 0 else 0.0))
    return out

def _similar(a,b):
    return SequenceMatcher(None, a, b).ratio()

def _save_debug_overlay(img, bbox, candidate, out_dir="/tmp", fname_prefix="debug"):
    """
    Save overlay showing bbox and candidate text for inspection.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        vis = img.copy()
        x,y,w,h = bbox
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(vis, candidate, (x, max(0,y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        path = os.path.join(out_dir, f"{fname_prefix}.png")
        _, buf = cv2.imencode('.png', vis)
        buf.tofile(path)
        return path
    except Exception:
        return None

def extract_waybill_code(image_path, debug=False):
    """
    Improved extraction: tries whole-image OCR + bottom-strip + barcode-based ROI extraction.
    Returns dict: {'extracted':..., 'raw':..., 'source':..., 'conf':..., ...}
    """

    orig, desk, th = preprocess_for_ocr(image_path) 
    h, w = desk.shape[:2]

    candidates = []


    try:
        imgs = [
            desk,
            cv2.resize(desk, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC),
            th,
            cv2.resize(th, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
        ]
        for im in imgs:
            ocr_res = run_all_engines(im)
            for r in ocr_res:
                raw = r.get('text', '')
                norm = normalize(raw)
                candidates.append({'raw': raw, 'norm': norm, 'conf': r.get('score', 0), 'src': 'ensemble_whole', 'bbox': None})
    except Exception:
        pass

    bottom_frac = 0.30
    y0 = int(h * (1.0 - bottom_frac))
    bottom = orig[y0:h, 0:w]
    bottom_gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)

    for psm in (6, 11, 7):
        try:
            lines = _tesseract_lines(bottom_gray, psm=psm)
            for txt, conf in lines:
                norm = normalize(txt)
                candidates.append({'raw': txt, 'norm': norm, 'conf': conf, 'src': f'tess_bottom_psm{psm}', 'bbox': (0, y0, w, h - y0)})
        except Exception:
            pass

    try:
        from .ocr_engine import reader as _ez_reader  
        ez = _ez_reader.readtext(bottom)
        for bbox, txt, score in ez:
            norm = normalize(txt)
            candidates.append({'raw': txt, 'norm': norm, 'conf': float(score), 'src': 'easy_bottom', 'bbox': (0, y0, w, h - y0)})
    except Exception:
        # fallback: call run_all_engines on bottom
        try:
            for r in run_all_engines(bottom):
                raw = r.get('text','')
                norm = normalize(raw)
                candidates.append({'raw': raw, 'norm': norm, 'conf': r.get('score',0), 'src': 'ensemble_bottom', 'bbox': (0, y0, w, h - y0)})
        except Exception:
            pass

    try:
        # detect horizontal dark bars using horizontal morphological filters
        g = cv2.GaussianBlur(desk, (3,3), 0)
        sob_y = cv2.Sobel(g, cv2.CV_8U, 0, 1, ksize=3)  # horizontal edges
        _, thsob = cv2.threshold(sob_y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # morphological close to join bars horizontally
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (40,4))
        morph = cv2.morphologyEx(thsob, cv2.MORPH_CLOSE, kern, iterations=2)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # keep wide-ish horizontal rectangles as barcode regions
        potential_barcodes = []
        for c in contours:
            x,y,wc,hc = cv2.boundingRect(c)
            if wc > 0.4 * w and hc > 6 and hc < 0.25*h:
                potential_barcodes.append((x,y,wc,hc))
        # sort by vertical position (top to bottom)
        potential_barcodes = sorted(potential_barcodes, key=lambda b: b[1])

        for (bx,by,bw,bh) in potential_barcodes[:3]:
            # the printed code often lies below the barcode; crop an ROI below barcode
            pad_y = int(bh * 1.8)
            roi_y0 = max(0, by + bh - 4)  # start a little inside barcode
            roi_y1 = min(h, by + bh + pad_y)
            roi = orig[roi_y0:roi_y1, max(0,bx-10): min(w, bx + bw + 10)]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # try multiple scales and psm modes
            for scale in (1.4, 1.8, 2.2):
                big = cv2.resize(roi_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                for psm in (7, 6, 11):
                    try:
                        lines = _tesseract_lines(big, psm=psm)
                        for txt, conf in lines:
                            norm = normalize(txt)
                            candidates.append({'raw': txt, 'norm': norm, 'conf': conf, 'src': f'roi_barcode_psm{psm}_s{scale}', 'bbox': (max(0,bx-10), roi_y0, min(w, bx + bw + 10) - max(0,bx-10), roi_y1 - roi_y0)})
                    except Exception:
                        pass
            # also run ensemble on roi
            try:
                for r in run_all_engines(roi):
                    raw = r.get('text','')
                    norm = normalize(raw)
                    candidates.append({'raw': raw, 'norm': norm, 'conf': r.get('score',0), 'src': 'ensemble_roi_barcode', 'bbox': (max(0,bx-10), roi_y0, min(w, bx + bw + 10) - max(0,bx-10), roi_y1 - roi_y0)})
            except Exception:
                pass
    except Exception:
        # barcode detection failed — continue
        pass

    uniq = {}
    for c in candidates:
        k = c['norm']
        if not k:
            continue
        # keep best conf for same norm
        if k not in uniq or c.get('conf',0) > uniq[k].get('conf',0):
            uniq[k] = c
    cand_list = list(uniq.values())

    # Strict pass first
    strict_matches = [c for c in cand_list if is_code(c['norm'])]
    if strict_matches:
        best = sorted(strict_matches, key=lambda x: (x.get('conf',0), len(x['norm'])), reverse=True)[0]
        if debug and best.get('bbox'):
            _save_debug_overlay(orig, best['bbox'], best['norm'], out_dir=os.path.join(os.path.dirname(image_path), "debug"), fname_prefix=os.path.splitext(os.path.basename(image_path))[0])
        return {'extracted': best['norm'], 'raw': best['raw'], 'conf': best.get('conf',0), 'src': best.get('src','strict')}

    # Fallback: candidates containing _1 or 1_ or ' 1 ' or long numeric+1+suffix combos
    fuzzy = []
    for c in cand_list:
        n = c['norm']
        if '_1' in n or '1_' in n or ' 1 ' in c['raw'] or (len(n) > 8 and '1' in n):
            score = c.get('conf',0) + 0.01*len(n)
            fuzzy.append((score, c))
    if fuzzy:
        fuzzy_sorted = sorted(fuzzy, key=lambda x: x[0], reverse=True)
        candidate = fuzzy_sorted[0][1]
        # attempt smart repair: if missing underscores, insert underscores around the '1'
        if not is_code(candidate['norm']):
            n = candidate['norm']
            # try find the single '1' occurrence and surround with underscores
            idx = n.find('1')
            if idx != -1:
                left = n[:idx]
                right = n[idx+1:]
                repaired = f"{left}_1_{right}"
                repaired = repaired.strip('_')
                repaired = normalize(repaired)
                if is_code(repaired):
                    return {'extracted': repaired, 'raw': candidate['raw'], 'conf': candidate.get('conf',0), 'src': candidate.get('src','fuzzy_repaired')}
        # final fallback return the candidate.norm (it might already be correct)
        if debug and candidate.get('bbox'):
            _save_debug_overlay(orig, candidate['bbox'], candidate['norm'], out_dir=os.path.join(os.path.dirname(image_path), "debug"), fname_prefix=os.path.splitext(os.path.basename(image_path))[0])
        return {'extracted': candidate['norm'], 'raw': candidate['raw'], 'conf': candidate.get('conf',0), 'src': candidate.get('src','fuzzy')}

    if debug:
        # save bottom strip image for manual inspection
        try:
            dpath = os.path.join(os.path.dirname(image_path), "debug")
            os.makedirs(dpath, exist_ok=True)
            _, buf = cv2.imencode('.jpg', bottom)
            buf.tofile(os.path.join(dpath, os.path.splitext(os.path.basename(image_path))[0] + "_bottom.jpg"))
        except Exception:
            pass

    return {'extracted': None, 'reason': 'no_candidate_found', 'candidates_tried': len(cand_list)}
