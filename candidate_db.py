
import os
import sqlite3
import numpy as np
import pyarrow.parquet as pq
import heapq

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator


PARQUET_PATH = (
    "data/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"
)

DATABASE_PATH = "data/candidate_database.sqlite"


BATCH_SIZE = 2048
FP_SIZE = 1024

fp_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=FP_SIZE
)


def build_candidate_database(
    parquet_path=PARQUET_PATH,
    database_path=DATABASE_PATH,
    batch_size=BATCH_SIZE
):
    # Create the database on disk
    os.makedirs(os.path.dirname(database_path), exist_ok=True)

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    # Create a table if it does not already exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            inchikey TEXT PRIMARY KEY,
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

    parquet_file = pq.ParquetFile(parquet_path)

    total_rows = 0
    inserted = 0
    invalid = 0

    # Read only the columns needed to build the database
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["inchikey", "normalized_smiles"]
    ):

        inchikeys = batch.column("inchikey").to_pylist()
        smiles_list = batch.column("normalized_smiles").to_pylist()

        for inchikey, smiles in zip(inchikeys, smiles_list):

            total_rows += 1

            if not inchikey or not smiles:
                invalid += 1
                continue

            # Skip molecules already in the database
            existing = cursor.execute(
                "SELECT 1 FROM candidates WHERE inchikey = ?",
                (inchikey,)
            ).fetchone()

            if existing is not None:
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

            # Convert fingerprint to 128-byte representation
            fp_array = np.zeros(FP_SIZE, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, fp_array)

            packed_fp = np.packbits(fp_array).tobytes()

            # Insert directly into SQLite
            cursor.execute("""
                INSERT OR IGNORE INTO candidates
                (inchikey, smiles, exact_mass, fingerprint)
                VALUES (?, ?, ?, ?)
            """, (
                inchikey,
                smiles,
                exact_mass,
                packed_fp
            ))

            inserted += cursor.rowcount

        # Commit periodically instead of after every molecule
        if total_rows % 10000 < batch.num_rows:
            connection.commit()

            print(
                f"Processed {total_rows:,} rows | "
                f"Unique molecules inserted: {inserted:,}"
            )

    connection.commit()

    # Report final database size
    count = cursor.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]

    connection.close()

    print("\nCandidate database created.")
    print(f"Rows processed: {total_rows:,}")
    print(f"Unique molecules: {count:,}")
    print(f"Invalid rows: {invalid:,}")

    return count


if __name__ == "__main__":
    build_candidate_database()
