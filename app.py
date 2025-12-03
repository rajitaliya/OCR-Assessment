import streamlit as st
from src.text_extraction import extract_waybill_code
from PIL import Image
import tempfile
import os


st.title("Waybill `_1_` Extractor (Open-source OCR ensemble)")

uploaded = st.file_uploader("Upload image", type=['png','jpg','jpeg','tiff'])

if uploaded:
    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, uploaded.name)

    # Save uploaded file safely
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.image(Image.open(path), caption="Uploaded image", use_column_width=True)

    if st.button("Run OCR"):
        with st.spinner("Processing..."):
            out = extract_waybill_code(path)
        st.json(out)
        if out.get("extracted"):
            st.success(f"Extracted: {out['extracted']}")
        else:
            st.error("No candidate matched the pattern.")