
import os
import sqlite3
import numpy as np
import pyarrow.parquet as pq
import heapq

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from config import TRAIN_DATA_PATH, TEST_DATA_PATH, DATABASE_PATH, BATCH_SIZE, FINGERPRINT_SIZE, ADDUCTS

PARQUET_PATH = (
    "data/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"
)

DATABASE_PATH = "data/candidate_database.sqlite"


BATCH_SIZE = 2048

fp_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=FINGERPRINT_SIZE,
)


# precursor_mz = (n * neutral_mass + mass_offset) / abs(charge)
def get_neutral_mass(precursor_mz, adduct):
    if adduct not in ADDUCTS:
        return precursor_mz  # Return the precursor m/z as a fallback for unknown adducts
        # raise ValueError(f"Unknown adduct: {adduct}")

    n, charge, mass_offset = ADDUCTS[adduct]

    return (
        abs(charge) * precursor_mz - mass_offset
    ) / n


def build_candidate_database(
    train_parquet_path=TRAIN_DATA_PATH,
    test_parquet_path=TEST_DATA_PATH,
    database_path=DATABASE_PATH,
    batch_size=BATCH_SIZE
):
    # Create the parent directory if necessary
    os.makedirs(os.path.dirname(os.path.abspath(database_path)), exist_ok=True)

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    try:
        # Create the candidate table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                inchikey14 TEXT PRIMARY KEY,
                smiles TEXT,
                exact_mass REAL,
                fingerprint BLOB
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_exact_mass
            ON candidates(exact_mass)
        """)

        connection.commit()

        # Check that an existing database has the expected schema
        columns = {
            row[1] for row in cursor.execute(
                "PRAGMA table_info(candidates)"
            ).fetchall()
        }

        if "inchikey14" not in columns:
            raise RuntimeError(
                "Existing database has an incompatible schema. "
                "Delete it and rebuild it."
            )

        # Process train and test files independently
        parquet_paths = [
            train_parquet_path,
            #test_parquet_path
        ]

        total_rows = 0
        inserted = 0
        invalid = 0

        for parquet_path in parquet_paths:

            parquet_file = pq.ParquetFile(parquet_path)

            required_columns = {
                "inchikey14",
                "normalized_smiles"
            }

            missing_columns = required_columns - set(
                parquet_file.schema_arrow.names
            )

            if missing_columns:
                raise ValueError(
                    f"Missing columns in {parquet_path}: "
                    f"{missing_columns}"
                )

            print(f"\nProcessing: {parquet_path}")

            for batch in parquet_file.iter_batches(
                batch_size=batch_size,
                columns=["inchikey14", "normalized_smiles"]
            ):

                inchikeys = batch.column("inchikey14").to_pylist()
                smiles_list = batch.column("normalized_smiles").to_pylist()

                for inchikey14, smiles in zip(inchikeys, smiles_list):

                    total_rows += 1

                    if not inchikey14 or not smiles:
                        invalid += 1
                        continue

                    # Parse molecular structure
                    mol = Chem.MolFromSmiles(smiles)

                    if mol is None:
                        invalid += 1
                        continue

                    # Calculate exact neutral molecular mass
                    exact_mass = Descriptors.ExactMolWt(mol)

                    # Generate Morgan fingerprint
                    fp = fp_generator.GetFingerprint(mol)

                    fp_array = np.zeros(
                        FINGERPRINT_SIZE,
                        dtype=np.uint8
                    )

                    DataStructs.ConvertToNumpyArray(
                        fp,
                        fp_array
                    )

                    packed_fp = np.packbits(fp_array).tobytes()

                    # Insert only if this connectivity key is not
                    # already present (e.g. in the train dataset)
                    cursor.execute("""
                        INSERT OR IGNORE INTO candidates
                        (inchikey14, smiles, exact_mass, fingerprint)
                        VALUES (?, ?, ?, ?)
                    """, (
                        inchikey14,
                        smiles,
                        exact_mass,
                        packed_fp
                    ))

                    inserted += cursor.rowcount

                # Commit once per batch
                connection.commit()

                if total_rows % 100000 < batch.num_rows:
                    print(
                        f"Processed {total_rows:,} rows | "
                        f"Unique molecules inserted: {inserted:,}"
                    )

        # Final database statistics
        count = cursor.execute(
            "SELECT COUNT(*) FROM candidates"
        ).fetchone()[0]

        print("\nCandidate database created.")
        print(f"Rows processed: {total_rows:,}")
        print(f"Unique molecules: {count:,}")
        print(f"New molecules inserted: {inserted:,}")
        print(f"Invalid rows: {invalid:,}")

        return count

    finally:
        connection.close()

if __name__ == "__main__":
    build_candidate_database()
