# Multilingual OCR

Multilingual OCR project for recognizing English, Hindi, Tamil, and Telugu text from word, line, and document images. The project uses a Vision Transformer backbone with a BiLSTM and CTC decoder, plus utilities for dataset loading, training, inference, document segmentation, and a Streamlit demo.

## Features

- OCR model architecture: ViT + BiLSTM + CTC
- Language-specific datasets for English, Hindi, Tamil, and Telugu
- Training and finetuning scripts for word and line OCR
- Multilingual inference router for English, Hindi, Tamil, and Telugu
- Document OCR pipeline with paragraph, line, and word segmentation
- Streamlit app for interactive OCR testing and metric display

## Project Structure

```txt
multilingual_ocr/
├── src/
│   ├── datasets/      # Dataset classes
│   ├── inference/     # Inference scripts
│   ├── models/        # ViT-BiLSTM-CTC model
│   ├── pipeline/      # Document OCR pipeline
│   ├── training/      # Training and finetuning scripts
│   └── utils/         # Text encoders, transforms, decoding helpers
├── fonts/             # Font files used by synthetic data scripts
├── streamlit_app.py   # Interactive OCR demo
├── requirements.txt   # Python dependencies
└── README.md
```

## Setup

Use a virtual environment so packages are installed locally for this project.

```bash
cd /Users/varungande/multilingual_ocr
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If VS Code still shows missing imports, select the project interpreter:

```txt
Cmd + Shift + P -> Python: Select Interpreter -> .venv/bin/python
```

## Required Local Artifacts

Model checkpoints and datasets are intentionally not committed to GitHub because they can be large. Place them in the expected local paths before running training or inference.

Expected model paths for the multilingual demo:

```txt
output/models/telugu/vit_lstm_te_best_handwritten_finetuned.pth
output/models/hindi/vit_lstm_hi_final.pth
output/models/tamil/vit_lstm_ta_best.pth
output/models/english/vit_lstm_en_cvl_finetuned.pth
```

Expected CVL English finetuning data:

```txt
data/cvl-database-1-1/trainset/words
data/cvl-database-1-1/testset/words
```

## Run the Streamlit Demo

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Upload a word or line image in the browser UI and choose either multilingual routing or a single-language model.

## Run Multilingual Inference

```bash
source .venv/bin/activate
python -m src.inference.infer_multilingual
```

When prompted, enter the path to an image file.

## Run Document OCR

```bash
source .venv/bin/activate
python -m src.pipeline.document_ocr "path/to/page_image.png"
```

Useful options:

```bash
python -m src.pipeline.document_ocr "path/to/page.png" --lang telugu
python -m src.pipeline.document_ocr "path/to/page.png" --mode handwritten
python -m src.pipeline.document_ocr "path/to/page.png" --no-save-crops
```

Generated OCR text, JSON, debug summaries, and crops are written to:

```txt
output/document_ocr/
```

## Train or Finetune

Example English CVL finetuning command:

```bash
source .venv/bin/activate
python -m src.training.finetune_english_cvl
```

Before running, make sure the CVL dataset and base checkpoint exist at the paths used in `src/training/finetune_english_cvl.py`.

## Notes

- This repository ignores `.venv/`, `data/`, `output/`, `metrics/`, and model checkpoint files.
- Use `requirements.txt` to recreate the Python environment.
- For macOS/Homebrew Python, install dependencies only inside the virtual environment. Do not use `--break-system-packages`.
