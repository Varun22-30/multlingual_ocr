import io
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.inference.infer_multilingual import (
    DEVICE,
    english_encoder,
    english_model,
    hindi_encoder,
    hindi_model,
    predict,
    select_multilingual_result,
    tamil_encoder,
    tamil_model,
    telugu_encoder,
    telugu_model,
)
from src.utils.image_transforms import ResizeAndPad
st.set_page_config(
    page_title="Multilingual OCR Demo",
    page_icon="OCR",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        max-width: 1200px;
    }
    div[data-testid="stFileUploader"] {
        max-width: 520px;
    }
    div[data-testid="stImage"] img {
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    div[data-testid="stHorizontalBlock"] {
        align-items: start;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


SUMMARY_FILES = {
    "English": os.path.join("metrics", "english_eval", "summary.txt"),
    "Hindi": os.path.join("metrics", "hindi_eval", "summary.txt"),
    "Tamil": os.path.join("metrics", "tamil_eval", "summary.txt"),
    "Telugu": os.path.join("metrics", "latest_eval", "summary.txt"),
    "Multilingual Router": os.path.join("metrics", "multilingual_eval_large", "summary.txt"),
}

CVL_ENGLISH_MODEL_PATH = os.path.join("output", "models", "english", "vit_lstm_en_cvl_finetuned.pth")


@st.cache_resource
def load_best_english_model():
    if os.path.exists(CVL_ENGLISH_MODEL_PATH):
        checkpoint = torch.load(CVL_ENGLISH_MODEL_PATH, map_location=DEVICE)
        num_classes = checkpoint["classifier.weight"].shape[0]
        from src.models.vit_bilstm_ctc import ViTBILSTMCTC

        model = ViTBILSTMCTC(num_classes=num_classes).to(DEVICE)
        model.load_state_dict(checkpoint)
        model.eval()
        return model, "CVL-Finetuned English"

    return english_model, "Base English"


best_english_model, best_english_label = load_best_english_model()

LANGUAGE_OPTIONS = {
    "English": ("en", best_english_model, english_encoder),
    "Hindi": ("hi", hindi_model, hindi_encoder),
    "Tamil": ("ta", tamil_model, tamil_encoder),
    "Telugu": ("te", telugu_model, telugu_encoder),
}


def parse_summary_file(path: str) -> Dict[str, float]:
    metrics = {}
    if not os.path.exists(path):
        return metrics

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r"^(.*?):\s*([0-9.]+)%?$", line.strip())
            if match:
                metrics[match.group(1).strip()] = float(match.group(2))
    return metrics


@st.cache_data
def load_metrics_table() -> pd.DataFrame:
    rows = []
    for model_name, path in SUMMARY_FILES.items():
        metrics = parse_summary_file(path)
        if not metrics:
            continue
        rows.append(
            {
                "Model": model_name,
                "Exact Match Accuracy": metrics.get("Exact Match Accuracy", 0.0),
                "Character Accuracy": metrics.get("Mean Character Accuracy", 0.0),
                "CER": metrics.get("Mean CER", 0.0),
                "WER": metrics.get("Mean WER", 0.0),
                "Language Detection Accuracy": metrics.get("Language Detection Accuracy", 0.0),
            }
        )
    return pd.DataFrame(rows)


def preprocess_by_lang(image: Image.Image, lang_code: str) -> torch.Tensor:
    if lang_code in {"hi", "ta"}:
        first_step = ResizeAndPad((224, 224))
    else:
        first_step = transforms.Resize((224, 224))

    pipeline = transforms.Compose([
        first_step,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return pipeline(image).unsqueeze(0)


def run_single_language_ocr(image: Image.Image, language_name: str) -> Dict[str, object]:
    lang_code, model, encoder = LANGUAGE_OPTIONS[language_name]
    tensor = preprocess_by_lang(image, lang_code)
    text, confidence = predict(model, encoder, tensor)
    return {
        "language": language_name,
        "text": text,
        "confidence": confidence,
    }


def run_multilingual_ocr(image: Image.Image) -> Dict[str, object]:
    image_hi = preprocess_by_lang(image, "hi")
    image_te = preprocess_by_lang(image, "te")
    image_ta = preprocess_by_lang(image, "ta")
    image_en = preprocess_by_lang(image, "en")

    text_hi, conf_hi = predict(hindi_model, hindi_encoder, image_hi)
    text_te, conf_te = predict(telugu_model, telugu_encoder, image_te)
    text_ta, conf_ta = predict(tamil_model, tamil_encoder, image_ta)
    text_en, conf_en = predict(best_english_model, english_encoder, image_en)

    scores = {
        "English": {"text": text_en, "confidence": conf_en},
        "Hindi": {"text": text_hi, "confidence": conf_hi},
        "Tamil": {"text": text_ta, "confidence": conf_ta},
        "Telugu": {"text": text_te, "confidence": conf_te},
    }

    result = select_multilingual_result(scores)
    return result


def make_metrics_plot(df: pd.DataFrame):
    labels = df["Model"].tolist()
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, df["Exact Match Accuracy"], marker="o", linewidth=2.2, label="Exact Match Accuracy")
    ax.plot(x, df["Character Accuracy"], marker="s", linewidth=2.2, label="Character Accuracy")
    ax.plot(x, df["CER"], marker="^", linewidth=2.2, label="CER")
    ax.plot(x, df["WER"], marker="D", linewidth=2.2, label="WER")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Overall OCR Performance Comparison")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    return fig


st.title("Multilingual OCR Project Demo")
st.caption(f"Interactive project presentation for English, Hindi, Tamil, Telugu OCR on `{DEVICE}`")

tab_demo, tab_metrics, tab_about = st.tabs(["OCR Demo", "Metrics Dashboard", "Project Overview"])


with tab_demo:
    st.subheader("Try OCR on an image")
    col_left, col_right = st.columns([1, 0.9], gap="large")

    with col_left:
        mode = st.radio("Inference Mode", ["Multilingual Router", "Single Language Model"], horizontal=True)
        uploaded_file = st.file_uploader(
            "Upload a word or line image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
        )
        if mode == "Single Language Model":
            selected_language = st.selectbox("Choose language", list(LANGUAGE_OPTIONS.keys()))
        else:
            selected_language = None

    with col_right:
        st.empty()

    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        preview_col, result_col = st.columns([0.55, 1.45], gap="large")

        with preview_col:
            st.markdown("**Uploaded Input**")
            st.image(image, width=260)

        with result_col:
            if st.button("Run OCR", type="primary", use_container_width=False):
                with st.spinner("Running OCR..."):
                    if mode == "Single Language Model":
                        result = run_single_language_ocr(image, selected_language)
                        c1, c2 = st.columns([1.25, 1])
                        with c1:
                            st.success(f"Predicted Text: {result['text'] or '[empty]'}")
                        with c2:
                            st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
                    else:
                        result = run_multilingual_ocr(image)
                        top1, top2 = st.columns([1.35, 1], gap="large")
                        with top1:
                            st.success(f"Predicted Text: {result['selected_text'] or '[empty]'}")
                            st.info(f"Detected Language: {result['selected_language']}")
                            st.caption(f"Decision reason: `{result['decision_reason']}`")
                        with top2:
                            best_conf = 0.0
                            if result["selected_language"] in result["scores"]:
                                best_conf = result["scores"][result["selected_language"]]["confidence"] * 100
                            st.metric("Selected Confidence", f"{best_conf:.2f}%")

                        details_df = pd.DataFrame(
                            [
                                {
                                    "Language": lang,
                                    "Prediction": payload["text"],
                                    "Confidence": round(payload["confidence"] * 100, 2),
                                    "Adjusted Score": round(result["adjusted_scores"][lang], 4),
                                    "Script Validity": round(result["validity"].get(lang, 0.0), 4),
                                }
                                for lang, payload in result["scores"].items()
                            ]
                        )
                        st.dataframe(details_df, use_container_width=True, hide_index=True)


with tab_metrics:
    st.subheader("Evaluation Metrics")
    metrics_df = load_metrics_table()

    if metrics_df.empty:
        st.warning("No evaluation summaries found yet. Run the evaluation scripts first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Exact Match", f"{metrics_df['Exact Match Accuracy'].max():.2f}%")
        c2.metric("Best Character Accuracy", f"{metrics_df['Character Accuracy'].max():.2f}%")
        c3.metric("Lowest CER", f"{metrics_df['CER'].min():.2f}%")

        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.pyplot(make_metrics_plot(metrics_df))

        if (metrics_df["Language Detection Accuracy"] > 0).any():
            st.subheader("Multilingual Routing Snapshot")
            lang_df = metrics_df[metrics_df["Language Detection Accuracy"] > 0][
                ["Model", "Language Detection Accuracy", "Exact Match Accuracy"]
            ]
            st.bar_chart(lang_df.set_index("Model"))


with tab_about:
    st.subheader("Project Summary")
    st.markdown(
        """
        This project is a multilingual OCR system built with:

        - Vision Transformer + BiLSTM + CTC architecture
        - Separate OCR models for English, Hindi, Tamil, and Telugu
        - A multilingual router that compares multiple model outputs
        - Handwritten Telugu support and evaluation tooling
        """
    )

    st.markdown("**Core demo highlights**")
    st.write("- Single-language OCR inference")
    st.write("- Multilingual language routing")
    st.write("- Metrics dashboard for report-ready results")
    st.write("- Streamlit presentation layer for demos and screenshots")

    st.markdown("**Current evaluated results**")
    metrics_df = load_metrics_table()
    if not metrics_df.empty:
        st.table(metrics_df[["Model", "Exact Match Accuracy", "Character Accuracy", "CER", "WER"]])
