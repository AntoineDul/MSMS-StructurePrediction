import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TRAIN_DATA_PATH, FINGERPRINT_SIZE, MAX_PEAKS, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE, POOLING_METHOD, DATASET_FRACTION
from DeepSets import DeepSets
from streamingDataset import StreamingMassSpecDataset

# Old collate function
"""
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
"""

# Improved collate function to include metadata for candidate retrieval
def collate_spectra(samples):
    # Remove invalid samples
    samples = [s for s in samples if s is not None]

    max_length = max(s["peaks"].shape[0] for s in samples)
    batch_size = len(samples)
    peak_dim = samples[0]["peaks"].shape[1]

    peaks = torch.zeros(batch_size, max_length, peak_dim)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

    for i, sample in enumerate(samples):
        n = sample["peaks"].shape[0]
        peaks[i, :n] = sample["peaks"]
        mask[i, :n] = True

    precursor = torch.stack([s["precursor"] for s in samples])
    targets = torch.stack([s["target"] for s in samples])

    # Metadata stays as lists
    inchikeys = [s["inchikey"] for s in samples]
    precursor_mzs = [s["precursor_mz"] for s in samples]
    adducts = [s["adduct"] for s in samples]

    return (
        peaks,
        mask,
        precursor,
        targets,
        inchikeys,
        precursor_mzs,
        adducts
    )


# Dataloaders for the Iterable Dataset implementation
def create_dataloaders():

    train_dataset = StreamingMassSpecDataset(
        parquet_path=TRAIN_DATA_PATH,
        split="train",
        batch_size=2048,
        max_peaks=MAX_PEAKS,
        shuffle_buffer_size=4096, 
    )

    val_dataset = StreamingMassSpecDataset(
        parquet_path=TRAIN_DATA_PATH,
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
def tanimoto_similarity(predictions, targets, threshold=0.2):

    # Convert probabilities into binary fingerprints
    predictions = (predictions >= threshold).float()
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

def soft_tanimoto_similarity(probabilities, targets):
    # Soft Tanimoto for the batch (using probabilities instead of binary predictions)
    soft_intersection = (probabilities * targets).sum(dim=1)

    soft_union = (
        probabilities.sum(dim=1)
        + targets.sum(dim=1)
        - soft_intersection
    )

    soft_tanimoto = (
        soft_intersection / soft_union.clamp(min=1e-8)
    )

    return soft_tanimoto.sum().item()

# Evaluate Tanimoto similarity at different thresholds
def evaluate_thresholds(predictions, targets):
    targets = targets.float()

    print("Threshold | Mean Tanimoto")
    print("------------------------")

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        binary_predictions = (predictions >= threshold).float()

        intersection = (binary_predictions * targets).sum(dim=1)
        union = (
            binary_predictions.sum(dim=1)
            + targets.sum(dim=1)
            - intersection
        )

        similarity = intersection / union.clamp(min=1e-8)
        similarity[union == 0] = 1.0

        print(f"{threshold:>9.1f} | {similarity.mean().item():.4f}")

def calculate_pos_weights(train_loader, device):

    positive_counts = torch.zeros(FINGERPRINT_SIZE)
    total_samples = 0

    for batch in tqdm(train_loader, desc="Calculating pos_weights"):

        targets = batch[3]

        positive_counts += targets.sum(dim=0)
        total_samples += targets.shape[0]

    negative_counts = total_samples - positive_counts

    # Avoid division by zero
    positive_counts = positive_counts.clamp(min=1)

    # Calculate negative / positive ratio for each bit
    weights = negative_counts / positive_counts

    # Use square root to avoid excessively large weights
    weights = torch.sqrt(weights)

    # Prevent rare bits from receiving enormous weights -> TODO : experiment with max values 5.0, 10.0...
    weights = weights.clamp(min=1.0, max=5.0)

    return weights.to(device)


# Train the model for one epoch 
def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for peaks, mask, precursor, targets, inchikeys, precursor_mzs, adducts in tqdm(
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
        # print(f"total_samples: {total_samples}")

    return total_loss / max(total_samples, 1), total_samples




@torch.no_grad()
def evaluate(model, loader, criterion, device, mean_fingerprint):

    model.eval()

    total_loss = 0.0
    total_tanimoto = 0.0
    total_soft_tanimoto = 0.0
    total_samples = 0

    for (
        peaks,
        mask,
        precursor,
        targets,
        inchikeys,
        precursor_mzs,
        adducts
    ) in tqdm(loader, desc="Validation"):

        peaks = peaks.to(device)
        mask = mask.to(device)
        precursor = precursor.to(device)
        targets = targets.to(device)

        logits = model(
            peaks,
            mask,
            precursor
        )

        loss = criterion(logits, targets)

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.2).float()

        tanimoto = tanimoto_similarity(predictions, targets, threshold=0.45)
        soft_tanimoto = soft_tanimoto_similarity(probabilities, targets)

        batch_size = peaks.size(0)

        total_loss += loss.item() * batch_size
        total_tanimoto += tanimoto
        total_soft_tanimoto += soft_tanimoto
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_tanimoto / total_samples,
        total_soft_tanimoto / total_samples
    )




# Evaluate the model on the validation set -> ISSUE: loads all predictions and targets into memory -> crash for large fraction of dataset
@torch.no_grad()
def evaluate_depracated(model, loader, criterion, device, mean_fingerprint):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_targets = []

    for peaks, mask, precursor, targets, inchikeys, precursor_mzs, adducts in tqdm(
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
        targets,
        0.2
    )

    baseline_predictions = mean_fingerprint.unsqueeze(0).expand(
        targets.shape[0], -1
    )

    print( "Baseline avg fingerprint Tanimoto similarity:")
    evaluate_thresholds(baseline_predictions, targets)
    
    print("Predictions Tanimoto similarity at different thresholds:")
    evaluate_thresholds(predictions, targets)
    
    print("Average active target bits:",
      targets.sum(dim=1).mean().item())

    print("Average predicted probability:",
        predictions.mean().item())

    print("Average predicted active bits at 0.2:",
        (predictions >= 0.2).sum(dim=1).float().mean().item())

    return avg_loss, avg_tanimoto


# Train the DeepSets model on the dataset
def train_model():

    total_samples = 0

    train_loader, val_loader = create_dataloaders()

    model = DeepSets(
        peak_dim=2,
        hidden_dim=128,
        fingerprint_size=FINGERPRINT_SIZE,
        pooling=POOLING_METHOD
    ).to(DEVICE)

    # Calculate positive weights for the loss function
    pos_weights = calculate_pos_weights(train_loader, DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    """
    # Compute mean fingerprint for the training set (baseline to compare results)
    bit_counts = torch.zeros(1024)
    num_samples = 0

    for batch in tqdm(train_loader, desc="Calculating mean fingerprint"):
        targets = batch[3]

        bit_counts += targets.sum(dim=0)
        num_samples += targets.shape[0]

    mean_fingerprint = bit_counts / num_samples
    """
    mean_fingerprint = 0

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_samples = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE
        )

        val_loss, val_tanimoto, val_soft_tanimoto = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE,
            mean_fingerprint
        )

        total_samples += train_samples

        print(f"Total training samples processed: {train_samples}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}")
        print(f"Val Tanimoto: {val_tanimoto:.4f}")
        print(f"Val Soft Tanimoto: {val_soft_tanimoto:.4f}")

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_tanimoto": val_tanimoto,
                    "val_soft_tanimoto": val_soft_tanimoto
                },
                f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_e{epoch+1}_df{DATASET_FRACTION}_deepsets.pt"
            )

            print("Best model saved!")

    return model


if __name__ == "__main__":
    model = train_model()

