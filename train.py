import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DATA_PATH, FINGERPRINT_SIZE, MAX_PEAKS, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE
from DeepSets import DeepSets
from streamingDataset import StreamingMassSpecDataset



# Pad batches of spectra to the same length and return a mask indicating which peaks are real vs padding
def collate_spectra(batch):

    peaks, precursors, fingerprints = zip(*batch)

    batch_size = len(batch)

    max_length = max(p.shape[0] for p in peaks)

    # Padded peak tensor
    padded_peaks = torch.zeros(
        batch_size,
        max_length,
        2
    )

    # True = real peak, False = padding
    mask = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.bool
    )

    for i, spectrum in enumerate(peaks):

        n = spectrum.shape[0]

        padded_peaks[i, :n] = spectrum
        mask[i, :n] = True

    precursors = torch.stack(precursors)
    fingerprints = torch.stack(fingerprints)

    return (
        padded_peaks,
        mask,
        precursors,
        fingerprints
    )


# Dataloaders for the Iterable Dataset implementation
def create_dataloaders():

    train_dataset = StreamingMassSpecDataset(
        parquet_path=DATA_PATH,
        split="train",
        batch_size=2048,
        max_peaks=MAX_PEAKS,
        shuffle_buffer_size=4096, 
    )

    val_dataset = StreamingMassSpecDataset(
        parquet_path=DATA_PATH,
        split="val",
        batch_size=2048,
        max_peaks=MAX_PEAKS,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_spectra,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_spectra,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    return train_loader, val_loader


# Calculate Tanimoto similarity between predicted and target fingerprints
def tanimoto_similarity(predictions, targets):

    # Convert probabilities into binary fingerprints
    predictions = (predictions >= 0.5).float()
    targets = targets.float()

    intersection = (predictions * targets).sum(dim=1)

    union = (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        - intersection
    )

    # Avoid division by zero
    similarity = intersection / union.clamp(min=1e-8)

    # Both fingerprints empty -> similarity 1
    similarity[union == 0] = 1.0

    return similarity.mean().item()

# Train the model for one epoch 
def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for peaks, mask, precursor, targets in tqdm(
        loader,
        desc="Training"
    ):

        peaks = peaks.to(device)
        mask = mask.to(device)
        precursor = precursor.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(peaks, mask, precursor)

        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = peaks.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


# Evaluate the model on the validation set
@torch.no_grad()
def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_targets = []

    for peaks, mask, precursor, targets in tqdm(
        loader,
        desc="Validation"
    ):

        peaks = peaks.to(device)
        mask = mask.to(device)
        precursor = precursor.to(device)
        targets = targets.to(device)

        logits = model(peaks, mask, precursor)

        loss = criterion(logits, targets)

        probabilities = torch.sigmoid(logits)

        batch_size = peaks.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_predictions.append(probabilities.cpu())
        all_targets.append(targets.cpu())

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    avg_loss = total_loss / max(total_samples, 1)

    avg_tanimoto = tanimoto_similarity(
        predictions,
        targets
    )

    return avg_loss, avg_tanimoto


# Train the DeepSets model on the dataset
def train_model():

    train_loader, val_loader = create_dataloaders()

    model = DeepSets(
        peak_dim=2,
        hidden_dim=128,
        fingerprint_size=FINGERPRINT_SIZE,
        pooling="sum"
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE
        )

        val_loss, val_tanimoto = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}")
        print(f"Val Tanimoto: {val_tanimoto:.4f}")

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_tanimoto": val_tanimoto
                },
                "best_deepsets.pt"
            )

            print("Best model saved!")

    return model


if __name__ == "__main__":
    model = train_model()











# Deprecated


"""
# Convert a SMILES string into a binary Morgan fingerprint
def smiles_to_fingerprint(smiles):

    # Get molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = fp_generator.GetFingerprint(mol)

    arr = np.zeros(FINGERPRINT_SIZE, dtype=np.float32)

    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr

# Generate one fingerprint per unique SMILES
def generate_fingerprints(df):

    unique_smiles = df["normalized_smiles"].unique()

    # Initialize dictionary to store fingerprints
    fingerprint_dict = {}

    # Iterate over SMILES and generate fingerprints
    for smiles in tqdm(unique_smiles, desc="Generating fingerprints"):

        fp = smiles_to_fingerprint(smiles)

        if fp is not None:
            fingerprint_dict[smiles] = fp

    # Remove molecules with invalid SMILES
    df = df[
        df["normalized_smiles"].isin(fingerprint_dict)
    ].reset_index(drop=True)

    return df, fingerprint_dict


# Load the relevant training columns from Parquet
def load_data(path):

    columns = [
        "normalized_smiles",
        "inchikey14",
        "precursor_mz",
        "ms2_mzs",
        "ms2_normalized_intensities"
    ]

    df = pd.read_parquet(path, columns=columns)

    # Remove rows with missing essential information
    df = df.dropna(subset=[
        "normalized_smiles",
        "inchikey14",
        "precursor_mz",
        "ms2_mzs",
        "ms2_normalized_intensities"
    ])

    # Remove invalid precursor masses
    df = df[df["precursor_mz"] > 0]

    return df.reset_index(drop=True)

    
# Split the dataset into training and validation sets ensuring that spectrums from the same molecule (ided by inchikey14) are NOT split across sets
def split_data(df, test_size=0.1):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=42
    )

    train_idx, val_idx = next(
        splitter.split(
            df,
            groups=df["inchikey14"]
        )
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df

        
        
"""
