import sqlite3
import numpy as np
import heapq
import torch
from tqdm import tqdm 

from DeepSets import DeepSets
from config import DEVICE, DATABASE_PATH, ADDUCTS
from train import create_dataloaders


# precursor_mz = (n * neutral_mass + mass_offset) / abs(charge)
def get_neutral_mass(precursor_mz, adduct):
    if adduct not in ADDUCTS:
        return precursor_mz  # Return the precursor m/z as a fallback for unknown adducts
        # raise ValueError(f"Unknown adduct: {adduct}")

    n, charge, mass_offset = ADDUCTS[adduct]

    return (
        abs(charge) * precursor_mz - mass_offset
    ) / n


def soft_tanimoto(predicted_fp, packed_candidate_fp):
    """
    Probability-weighted Tanimoto similarity.

    predicted_fp: array of 1024 probabilities
    packed_candidate_fp: 128-byte packed Morgan fingerprint
    """

    candidate_fp = np.unpackbits(
        np.frombuffer(packed_candidate_fp, dtype=np.uint8)
    ).astype(np.float32)

    intersection = np.dot(predicted_fp, candidate_fp)

    union = (
        predicted_fp.sum()
        + candidate_fp.sum()
        - intersection
    )

    return float(intersection / union) if union > 0 else 0.0


def retrieve_and_rank(
    cursor,
    predicted_fp,
    neutral_mass,
    true_inchikey,
    mass_tolerance_ppm=10,
    top_k=25,
    fetch_size=1000
):
    """
    Retrieve candidates from SQLite and rank them.

    Returns:
        top_candidates: top-k ranked candidates
        true_rank: rank of the ground-truth molecule, or None
        candidate_count: number of mass-compatible candidates
    """

    tolerance = neutral_mass * mass_tolerance_ppm / 1e6

    min_mass = neutral_mass - tolerance
    max_mass = neutral_mass + tolerance

    # Find the ground-truth molecule in the database.
    cursor.execute("""
        SELECT exact_mass, fingerprint
        FROM candidates
        WHERE inchikey = ?
    """, (true_inchikey,))

    target = cursor.fetchone()

    if target is None:
        return [], None, 0

    target_mass, target_fp = target

    # Check whether the correct molecule survives mass filtering.
    if not (min_mass <= target_mass <= max_mass):
        return [], None, 0

    target_score = soft_tanimoto(predicted_fp, target_fp)

    # Query only mass-compatible candidates.
    cursor.execute("""
        SELECT inchikey, smiles, exact_mass, fingerprint
        FROM candidates
        WHERE exact_mass BETWEEN ? AND ?
    """, (min_mass, max_mass))

    # Keep only the best top-k candidates in memory.
    best_candidates = []

    candidate_count = 0
    better_than_target = 0

    while True:
        rows = cursor.fetchmany(fetch_size)

        if not rows:
            break

        for inchikey, smiles, exact_mass, fingerprint in rows:

            candidate_count += 1

            score = soft_tanimoto(
                predicted_fp,
                fingerprint
            )

            # Ranking order: higher score first, then
            # lexicographically larger InChIKey for exact ties.
            if (
                score > target_score
                or (
                    score == target_score
                    and inchikey > true_inchikey
                )
            ):
                better_than_target += 1

            result = {
                "inchikey": inchikey,
                "smiles": smiles,
                "exact_mass": exact_mass,
                "similarity": score
            }

            item = (score, inchikey, result)

            if len(best_candidates) < top_k:
                heapq.heappush(best_candidates, item)

            elif item[:2] > best_candidates[0][:2]:
                heapq.heapreplace(best_candidates, item)

    true_rank = better_than_target + 1

    top_candidates = [
        item[2]
        for item in sorted(best_candidates, reverse=True)
    ]

    return top_candidates, true_rank, candidate_count



def evaluate_retrieval(
    model,
    val_loader,
    database_path=DATABASE_PATH,
    mass_tolerance_ppm=10,
    top_k=25,
    device="cpu"
):

    model.eval()

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    total = 0
    top1_correct = 0
    top25_correct = 0
    reciprocal_rank_sum = 0.0

    missing_candidates = 0

    with torch.no_grad():

        for batch in tqdm(val_loader, desc="Retrieval evaluation"):

            (
                peaks,
                mask,
                precursor,
                targets,
                inchikeys,
                precursor_mzs,
                adducts
            ) = batch

            peaks = peaks.to(device)
            mask = mask.to(device)
            precursor = precursor.to(device)

            # Predict fingerprints for the entire batch
            logits = model(peaks, mask, precursor)

            probabilities = torch.sigmoid(logits).cpu().numpy()

            # Retrieve candidates for each spectrum
            for i in range(len(inchikeys)):

                total += 1

                true_inchikey = inchikeys[i]

                # Convert precursor m/z to neutral mass
                try:
                    neutral_mass = get_neutral_mass(
                        precursor_mzs[i],
                        adducts[i]
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Could not evaluate spectrum with "
                        f"adduct {adducts[i]}"
                    ) from error

                # Retrieve and rank candidates
                candidates, rank, candidate_count = (
                    retrieve_and_rank(
                        cursor=cursor,
                        predicted_fp=probabilities[i],
                        neutral_mass=neutral_mass,
                        true_inchikey=true_inchikey,
                        mass_tolerance_ppm=mass_tolerance_ppm,
                        top_k=top_k
                    )
                )

                if rank is None:
                    missing_candidates += 1
                    continue

                # Top-1 accuracy
                if rank == 1:
                    top1_correct += 1

                # Top-25 accuracy
                if rank <= 25:
                    top25_correct += 1

                # MRR@25
                if rank <= 25:
                    reciprocal_rank_sum += 1.0 / rank

    connection.close()

    # Calculate final metrics
    top1_accuracy = top1_correct / total
    top25_accuracy = top25_correct / total
    mrr_at_25 = reciprocal_rank_sum / total

    candidate_coverage = (
        (total - missing_candidates) / total
    )

    print("\n========== RETRIEVAL RESULTS ==========")

    print(f"Evaluated spectra: {total:,}")
    print(f"Missing candidates: {missing_candidates:,}")
    print(f"Candidate coverage: {candidate_coverage:.4f}")

    print(f"\nTop-1 accuracy:  {top1_accuracy:.4f}")
    print(f"Top-25 accuracy: {top25_accuracy:.4f}")
    print(f"MRR@25:          {mrr_at_25:.4f}")

    print("=======================================")

    return {
        "top1_accuracy": top1_accuracy,
        "top25_accuracy": top25_accuracy,
        "mrr_at_25": mrr_at_25,
        "candidate_coverage": candidate_coverage,
        "evaluated_spectra": total
    }


if __name__ == "__main__":


    train_loader, val_loader = create_dataloaders()

    model = DeepSets().to(DEVICE)

    checkpoint = torch.load(
        "best_deepsets.pt",
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"Validation Tanimoto: {checkpoint['val_tanimoto']:.4f}")

    results = evaluate_retrieval(
        model=model,
        val_loader=val_loader,
        database_path=DATABASE_PATH,
        mass_tolerance_ppm=10,
        top_k=25,
        device=DEVICE
    )
