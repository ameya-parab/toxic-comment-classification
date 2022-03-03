import os

import torch

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")
CACHE_DIR = os.path.join(ROOT_DIR, ".huggingface_cache")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

STORAGE = f"sqlite:///{os.path.join(ROOT_DIR, 'study', 'hyperparameter_studies.db')}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LEN = 510
