# Smart Shelf Guard

AI-Powered Supermarket Expiry Warning App for PARKnSHOP.

## Overview

Smart Shelf Guard helps supermarket staff identify high-risk and expired products by analyzing product images through three deep learning pipelines:

- **Pipeline 1 — Shelf-life Classification:** Fine-tuned Swin-tiny classifies products into short shelf-life, medium shelf-life, or non-perishable.
- **Pipeline 2 — Freshness Detection:** Fine-tuned ViT-base detects whether short shelf-life products are fresh or rotten. Only activates for short shelf-life items.
- **Pipeline 3 — Image Captioning:** Pre-trained BLIP-large generates a text description of the product image.

A business logic layer combines model outputs with user inputs (entry date, shelf life, seasonal month) to produce a risk score (0–100) with actionable recommendations.

## Live Demo

- **Streamlit Cloud:** https://group24project.streamlit.app/
- **HuggingFace Spaces (backup):** https://huggingface.co/spaces/Alisa-Sun/smart-shelf-guard

## Setup

### Requirements

- Python 3.10+
- ~4GB RAM (for loading three models)
- GPU optional (speeds up inference but not required)

### Installation

```bash
git clone https://github.com/alisasun2025/Group24_program.git
cd Group24_program
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Usage

1. Upload a product image using the file uploader.
2. Set the warehouse entry date and shelf-life duration in the sidebar.
3. Adjust the seasonal month if needed.
4. View the risk assessment: category, freshness, description, risk score, and recommendation.

## Models

| Pipeline | Model | HuggingFace |
|----------|-------|-------------|
| 1 | Swin-tiny (fine-tuned) | [Alisa-Sun/shelf-life-classification](https://huggingface.co/Alisa-Sun/shelf-life-classification) |
| 2 | ViT-base (fine-tuned) | [Alisa-Sun/freshness-detection](https://huggingface.co/Alisa-Sun/freshness-detection) |
| 3 | BLIP-large (pre-trained) | [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large) |

## Risk Score

| Category | Formula | Max |
|----------|---------|-----|
| Short shelf-life | Time (0-50) + Freshness (0-30) + Season (0-20) | 100 |
| Medium shelf-life | Time (0-80) + Season (0-20) | 100 |
| Non-perishable | Always 0 | 0 |
| Expired | Displayed as EXPIRED | — |

## Project Structure

```
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
└── README.md
```

## Team

**Group 24 — ISOM5240**
- SUN, Lijia (21192595)
- LIU, Jiawen (21232084)
