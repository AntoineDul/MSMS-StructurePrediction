import torch

DATA_PATH = "data/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"
DATABASE_PATH = "data/candidate_database.sqlite"

FINGERPRINT_SIZE = 1024
FINGERPRINT_RADIUS = 2
LRU_CACHE_SIZE = 50000

MAX_PEAKS = 200

BATCH_SIZE = 256
DATASET_FRACTION = 1.0
EPOCHS = 10
LEARNING_RATE = 1e-3
POOLING_METHOD = "mean"  # Options: "mean", "sum"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

# Format: adduct: (number_of_molecules, charge, mass_offset)
ADDUCTS = {
    # Protonated / deprotonated molecules
    "[M+H]+":           (1,  1,   1.007276),
    "[M-H]-":           (1, -1,  -1.007276),
    "[M]+":             (1,  1,   0.0),
    "[M]-":             (1, -1,   0.0),

    # Metal adducts
    "[M+Na]+":          (1,  1,  22.989218),
    "[M+K]+":           (1,  1,  38.963158),
    "[M+Li]+":          (1,  1,   7.015455),
    "[M+Ca]2+":         (1,  2,  39.961494),
    "[M+NH4]+":         (1,  1,  18.033823),

    # Multiple charges
    "[M+2H]2+":         (1,  2,   2 * 1.007276),
    "[M+3H]3+":         (1,  3,   3 * 1.007276),
    "[M-2H]2-":         (1, -2,  -2 * 1.007276),

    # Common neutral losses
    "[M-H2O+H]+":       (1,  1, -17.003289),
    "[M-2H2O+H]+":      (1,  1, -35.013854),

    # Other common adducts
    "[M+Cl]-":          (1, -1,  34.969402),

    # Formate and acetate adducts
    "[M+CH2O2-H]-":     (1, -1,  44.998201),
    "[M+C2H4O2-H]-":    (1, -1,  59.013851),

    # Dimers
    "[2M+H]+":          (2,  1,   1.007276),
    "[2M+Na]+":         (2,  1,  22.989218),
    "[2M-H]-":          (2, -1,  -1.007276),
    "[2M+NH4]+":        (2,  1,  18.033823),
    "[2M+K]+":          (2,  1,  38.963158),
}