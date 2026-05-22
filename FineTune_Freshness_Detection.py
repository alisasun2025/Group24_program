# =============================================================
# ISOM5240 Fine-tuning Notebook
# Pipeline 2: Product Freshness Detection
# =============================================================
# Workflow:
#   Phase 1 → Evaluate 3 pre-trained models (NO fine-tuning)
#   Phase 2 → Select the best one
#   Phase 3 → Fine-tune ONLY the selected model
#   Phase 4 → Compare before vs after fine-tuning
#   Phase 5 → Push fine-tuned model to HuggingFace Hub
# =============================================================

# %% [markdown]
# ## Step 1: Install dependencies

# %%
# !pip install transformers datasets evaluate accelerate pillow -q
# !pip install huggingface_hub -q

# %% [markdown]
# ## Step 2: Login to HuggingFace Hub

# %%
# from huggingface_hub import notebook_login
# notebook_login()

# %% [markdown]
# ## Step 3: Load and prepare dataset
#
# Dataset: Fresh and Rotten Fruits
# Original classes: fresh_apple, fresh_banana, fresh_orange,
#                   rotten_apple, rotten_banana, rotten_orange
# Re-label to binary: fresh (0) / rotten (1)

# %%
from datasets import load_dataset
import numpy as np
import time

# Option A: From HuggingFace
# dataset = load_dataset("Densu341/Fresh-rotten-fruit")

# Option B: From local imagefolder (Kaggle download)
# dataset = load_dataset("imagefolder", data_dir="dataset/")

# Re-label to binary
def relabel_to_binary(example):
    original_names = dataset["train"].features["label"].names
    original_label = original_names[example["label"]]
    example["label"] = 0 if "fresh" in original_label.lower() else 1
    return example

LABEL_NAMES = ["fresh", "rotten"]

# dataset = dataset.map(relabel_to_binary)
# print(f"Train: {len(dataset['train'])} | Test: {len(dataset['test'])}")


# ==============================================================
# PHASE 1: Evaluate 3 pre-trained models (NO fine-tuning)
# ==============================================================

# %% [markdown]
# ## Step 4: Define candidate models

# %%
CANDIDATE_MODELS = {
    "ViT-base": "google/vit-base-patch16-224",
    "ResNet-50": "microsoft/resnet-50",
    "Swin-tiny": "microsoft/swin-tiny-patch4-window7-224",
}

# %% [markdown]
# ## Step 5: Evaluate each pre-trained model on test set (zero-shot)
#
# Since these are ImageNet-pretrained models, they output ImageNet labels
# (e.g., "banana", "orange"), not "fresh"/"rotten" directly.
# We use a keyword-based mapping to approximate freshness classification
# and measure how well each model's features transfer to this task.

# %%
from transformers import pipeline as hf_pipeline
import pandas as pd

def evaluate_pretrained(model_key, model_path, test_data, num_samples=200):
    """
    Evaluate a pre-trained model WITHOUT fine-tuning.
    Measures: inference speed and feature quality (via confidence distribution).
    """
    print(f"\nEvaluating: {model_key} ({model_path})")

    pipe = hf_pipeline("image-classification", model=model_path)

    # Count parameters
    total_params = sum(p.numel() for p in pipe.model.parameters())

    # Run inference on test samples
    n = min(num_samples, len(test_data))
    inference_times = []
    confidences = []

    for i in range(n):
        img = test_data[i]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        t0 = time.time()
        result = pipe(img, top_k=1)
        inference_times.append(time.time() - t0)
        confidences.append(result[0]["score"])

    avg_time_ms = np.mean(inference_times) * 1000
    avg_confidence = np.mean(confidences)

    result = {
        "Model": model_key,
        "Parameters": f"{total_params:,}",
        "Avg Inference (ms)": round(avg_time_ms, 1),
        "Avg Confidence": round(avg_confidence, 4),
        "Samples Tested": n,
    }

    print(f"  Params: {total_params:,} | Speed: {avg_time_ms:.1f}ms | Conf: {avg_confidence:.4f}")
    return result

# %%
# Run Phase 1 evaluation
pretrained_results = []

# TODO: Uncomment when dataset is ready
# for key, path in CANDIDATE_MODELS.items():
#     r = evaluate_pretrained(key, path, dataset["test"])
#     pretrained_results.append(r)
#
# df_pretrained = pd.DataFrame(pretrained_results)
# print("\n" + "="*60)
# print("PHASE 1 RESULTS: Pre-trained Model Comparison")
# print("="*60)
# print(df_pretrained.to_string(index=False))


# %% [markdown]
# ## Step 6: Select the best model
#
# Selection criteria:
# - Primary: which model's features are most relevant (highest confidence
#   on correct categories, lowest inference time)
# - Secondary: parameter count (smaller = faster to fine-tune)
#
# Based on typical results:
# - ViT-base: highest accuracy after fine-tuning, moderate speed
# - ResNet-50: fastest inference, good accuracy
# - Swin-tiny: good balance, slightly better than ResNet

# %%
# TODO: Update this based on your actual Phase 1 results
SELECTED_MODEL_KEY = "ViT-base"  # Change after comparing
SELECTED_MODEL_PATH = CANDIDATE_MODELS[SELECTED_MODEL_KEY]

print(f"Selected model for fine-tuning: {SELECTED_MODEL_KEY}")
print(f"Path: {SELECTED_MODEL_PATH}")


# ==============================================================
# PHASE 2: Fine-tune the selected model
# ==============================================================

# %% [markdown]
# ## Step 7: Preprocessing for the selected model

# %%
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained(SELECTED_MODEL_PATH)

def preprocess(example):
    image = example["image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    example["pixel_values"] = inputs["pixel_values"].squeeze(0)
    return example

# train_processed = dataset["train"].map(preprocess, remove_columns=["image"])
# test_processed = dataset["test"].map(preprocess, remove_columns=["image"])
# print(f"Preprocessing done with {SELECTED_MODEL_KEY} processor")


# %% [markdown]
# ## Step 8: Load model with custom classification head

# %%
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained(
    SELECTED_MODEL_PATH,
    num_labels=len(LABEL_NAMES),
    id2label={i: l for i, l in enumerate(LABEL_NAMES)},
    label2id={l: i for i, l in enumerate(LABEL_NAMES)},
    ignore_mismatched_sizes=True,
)

total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {SELECTED_MODEL_KEY}")
print(f"Total params: {total_params:,} | Trainable: {trainable:,}")


# %% [markdown]
# ## Step 9: Training configuration

# %%
import evaluate
from transformers import TrainingArguments, Trainer

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir=f"./freshness-{SELECTED_MODEL_KEY.lower()}",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
    remove_unused_columns=False,
    push_to_hub=False,  # We push manually in Step 12
)

print("Training args configured: 5 epochs, lr=2e-5, batch=16")


# %% [markdown]
# ## Step 10: Train!

# %%
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_processed,
#     eval_dataset=test_processed,
#     compute_metrics=compute_metrics,
# )
#
# train_start = time.time()
# trainer.train()
# train_time = time.time() - train_start
#
# print(f"\nTraining complete in {train_time/60:.1f} minutes")


# %% [markdown]
# ## Step 11: Evaluate fine-tuned model + compare with pre-trained

# %%
# # Fine-tuned accuracy
# ft_results = trainer.evaluate()
# print(f"Fine-tuned Accuracy: {ft_results['eval_accuracy']:.4f}")
# print(f"Fine-tuned Loss: {ft_results['eval_loss']:.4f}")
#
# # Inference speed after fine-tuning
# pipe_ft = hf_pipeline("image-classification", model=model, image_processor=processor)
# ft_times = []
# for i in range(min(100, len(dataset["test"]))):
#     img = dataset["test"][i]["image"]
#     if img.mode != "RGB":
#         img = img.convert("RGB")
#     t0 = time.time()
#     pipe_ft(img)
#     ft_times.append(time.time() - t0)
# ft_avg_ms = np.mean(ft_times) * 1000
#
# # Summary table: Pre-trained vs Fine-tuned
# comparison = pd.DataFrame([
#     {
#         "Stage": "Pre-trained (before)",
#         "Model": SELECTED_MODEL_KEY,
#         "Accuracy": "N/A (ImageNet labels)",
#         "Avg Inference (ms)": pretrained_results[...]["Avg Inference (ms)"],
#     },
#     {
#         "Stage": "Fine-tuned (after)",
#         "Model": SELECTED_MODEL_KEY,
#         "Accuracy": f"{ft_results['eval_accuracy']:.4f}",
#         "Avg Inference (ms)": round(ft_avg_ms, 1),
#     },
# ])
# print("\n" + "="*60)
# print("PHASE 2 RESULTS: Before vs After Fine-tuning")
# print("="*60)
# print(comparison.to_string(index=False))


# %% [markdown]
# ## Step 12: Push fine-tuned model to HuggingFace Hub

# %%
HUB_MODEL_ID = "your-username/freshness-detection"  # TODO: Change

# trainer.push_to_hub(
#     repo_id=HUB_MODEL_ID,
#     commit_message=f"Fine-tuned {SELECTED_MODEL_KEY} for freshness detection"
# )
# processor.push_to_hub(HUB_MODEL_ID)
# print(f"Model pushed to: https://huggingface.co/{HUB_MODEL_ID}")


# %% [markdown]
# ## Step 13: Final inference test

# %%
# from PIL import Image
# pipe = hf_pipeline("image-classification", model=HUB_MODEL_ID)
#
# test_img = Image.open("test_rotten_banana.jpg")
# print(pipe(test_img))
# # Expected: [{'label': 'rotten', 'score': 0.96}, {'label': 'fresh', 'score': 0.04}]


# %% [markdown]
# ## Final Experiment Table (for Excel submission)
#
# ### Table 1: Model Selection (Phase 1 — pre-trained comparison)
# | Model | Parameters | Avg Inference (ms) | Avg Confidence | Selected? |
# |-------|-----------|-------------------|----------------|-----------|
# | ViT-base | 86M | 45.2 | 0.82 | ✅ |
# | ResNet-50 | 25M | 31.8 | 0.78 | |
# | Swin-tiny | 28M | 37.9 | 0.80 | |
#
# ### Table 2: Fine-tuning Result (Phase 2)
# | Stage | Model | Accuracy | Loss | Inference (ms) |
# |-------|-------|----------|------|----------------|
# | Pre-trained | ViT-base | N/A | N/A | 45.2 |
# | Fine-tuned | ViT-base | 0.96 | 0.12 | 46.1 |
#
# (Replace with actual numbers)
