import sqlite3
import numpy as np
import heapq

DATABASE_PATH = "data/candidate_database.sqlite"

ADDUCT_MASS_OFFSETS = {
    "[M+H]+": 1.007276,
    "[M+Na]+": 22.989218,
    "[M-H]-": -1.007276,
    "[M+K]+": 39.098342,
}



### Candidate prediction functions ###
 
def get_neutral_mass(precursor_mz, adduct, charge=1):
    offset = ADDUCT_MASS_OFFSETS[adduct]
    return precursor_mz * abs(charge) - offset


def retrieve_candidates_from_db(
    predicted_probabilities,
    neutral_mass,
    database_path=DATABASE_PATH,
    mass_tolerance_ppm=10,
    top_k=25,
    fetch_size=1000
):
    """
    Rank database candidates using predicted fingerprint
    probabilities and precursor neutral mass.

    predicted_probabilities: numpy array of shape (1024,)
    neutral_mass: neutral molecular mass in Da
    """

    tolerance = neutral_mass * mass_tolerance_ppm / 1e6

    min_mass = neutral_mass - tolerance
    max_mass = neutral_mass + tolerance

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    # Retrieve only candidates in the allowed mass range
    cursor.execute("""
        SELECT inchikey, smiles, exact_mass, fingerprint
        FROM candidates
        WHERE exact_mass BETWEEN ? AND ?
    """, (min_mass, max_mass))

    # Keep only the top-k results
    best_candidates = []

    while True:
        rows = cursor.fetchmany(fetch_size)

        if not rows:
            break

        for inchikey, smiles, exact_mass, packed_fp in rows:

            # Unpack the 1024-bit fingerprint
            candidate_fp = np.unpackbits(
                np.frombuffer(packed_fp, dtype=np.uint8)
            ).astype(np.float32)

            # Probability-weighted Tanimoto similarity
            intersection = np.sum(
                predicted_probabilities * candidate_fp
            )

            union = (
                np.sum(predicted_probabilities)
                + np.sum(candidate_fp)
                - intersection
            )

            similarity = (
                intersection / union if union > 0 else 0.0
            )

            result = {
                "inchikey": inchikey,
                "smiles": smiles,
                "exact_mass": exact_mass,
                "similarity": float(similarity)
            }

            # Maintain a min-heap of the best top-k candidates
            item = (similarity, inchikey, result)

            if len(best_candidates) < top_k:
                heapq.heappush(best_candidates, item)
            elif similarity > best_candidates[0][0]:
                heapq.heapreplace(best_candidates, item)

    connection.close()

    # Highest similarity first
    results = [
        item[2]
        for item in sorted(best_candidates, reverse=True)
    ]

    return results