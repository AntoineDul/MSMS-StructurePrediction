import torch

DATA_PATH = "data/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"

FINGERPRINT_SIZE = 1024
FINGERPRINT_RADIUS = 2
LRU_CACHE_SIZE = 50000

MAX_PEAKS = 200

BATCH_SIZE = 256
DATASET_FRACTION = 0.25
EPOCHS = 5
LEARNING_RATE = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)