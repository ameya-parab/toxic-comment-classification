import os

import torch

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
STUDY_DIR = os.path.join(ROOT_DIR, "study")
CACHE_DIR = os.path.join(ROOT_DIR, ".huggingface_cache")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(STUDY_DIR, exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

STORAGE = f"sqlite:///{os.path.join(ROOT_DIR, 'study', 'hyperparameter_studies.db')}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CHECKPOINT = "distilbert-base-uncased"

METRICS = "hamming_loss"

MAX_LEN = 510

THRESHOLD = 0.5
