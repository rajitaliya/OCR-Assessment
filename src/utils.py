import re
import difflib

PATTERN = re.compile(r'\b[A-Za-z0-9]+_1_[A-Za-z0-9]+\b')

# common confusion map
CONFUSION_MAP = str.maketrans({
    'I':'1', 'l':'1', '|':'1', '!':'1', 'O':'0', 'o':'0', 'S':'5', 's':'5'
})

def normalize_candidate(text):
    t = text.strip()
    t = t.replace(' ', '')  # remove spaces inside codes
    t = t.translate(CONFUSION_MAP)
    return t

def matches_pattern(text):
    return bool(PATTERN.search(text))

def extract_candidates(ocr_lines):
    # ocr_lines: list of dicts with 'text', 'score', ...
    candidates = []
    for item in ocr_lines:
        norm = normalize_candidate(item['text'])
        if matches_pattern(norm):
            candidates.append({**item, 'norm': norm})
    return candidates

def fuzzy_score(target, candidate):
    # quick similarity ratio
    return difflib.SequenceMatcher(None, target, candidate).ratio()

def choose_best_candidate(candidates):
    # pick highest score (engine score) and/or longest length
    if not candidates:
        return None
    candidates_sorted = sorted(candidates, key=lambda x: (x.get('score',0), len(x['norm'])), reverse=True)
    return candidates_sorted[0]
