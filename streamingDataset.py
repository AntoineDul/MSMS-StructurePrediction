import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq
import hashlib
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from functools import lru_cache

from config import FINGERPRINT_SIZE, FINGERPRINT_RADIUS, DATASET_FRACTION, LRU_CACHE_SIZE

fp_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=FINGERPRINT_RADIUS,
    fpSize=FINGERPRINT_SIZE
)

# Determine if a molecule should be included in the training subset
def include_in_subset(inchikey, fraction=DATASET_FRACTION):
    digest = hashlib.blake2b(
        str(inchikey).encode(),
        digest_size=8
    ).digest()

    value = int.from_bytes(digest, "big")

    return value % 100 < int(fraction * 100)


# Cache fingerprints to avoid recomputation (increases speed significantly)
@lru_cache(maxsize=LRU_CACHE_SIZE)
def cached_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = fp_generator.GetFingerprint(mol)

    arr = np.zeros(1024, dtype=np.uint8)

    DataStructs.ConvertToNumpyArray(fp, arr)

    return np.packbits(arr).tobytes()


# Convert a SMILES string into a binary Morgan fingerprint
def get_fingerprint(smiles):

    packed = cached_fingerprint(smiles)

    if packed is None:
        return None

    return np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8)
    ).astype(np.float32)


# Deterministic split
def get_split(inchikey):
    """
    Assign all spectra belonging to a molecule
    to the same split.
    """

    digest = hashlib.blake2b(
        str(inchikey).encode(),
        digest_size=8
    ).digest()

    value = int.from_bytes(digest, byteorder="big")

    return "val" if value % 10 == 0 else "train"


# Iterarble dataset implementation for handling the very large train dataset
class StreamingMassSpecDataset(IterableDataset):

    def __init__(
        self,
        parquet_path,
        split="train",
        batch_size=2048,
        max_peaks=200,
        shuffle_buffer_size=4096,
    ):

        self.parquet_path = parquet_path
        self.split = split
        self.batch_size = batch_size
        self.max_peaks = max_peaks

        self.shuffle_buffer_size = (
            shuffle_buffer_size if split == "train" else 0
        )

    def process_row(self, smiles, inchikey, precursor_mz, mzs, intensities):

        # Validate row
        if (
            smiles is None
            or inchikey is None
            or precursor_mz is None
            or mzs is None
            or intensities is None
        ):
            return None

        if not np.isfinite(precursor_mz) or precursor_mz <= 0:
            return None

        # Assign molecule to train or validation
        if get_split(inchikey) != self.split:
            return None

        if self.split == "train":
            if not include_in_subset(inchikey, fraction=DATASET_FRACTION):
                return None

        # Generate target fingerprint
        fingerprint = get_fingerprint(smiles)

        if fingerprint is None:
            return None

        mzs = np.asarray(mzs, dtype=np.float32)
        intensities = np.asarray(intensities, dtype=np.float32)

        # Keep corresponding, valid peaks
        n = min(len(mzs), len(intensities))

        mzs = mzs[:n]
        intensities = intensities[:n]

        valid = (
            np.isfinite(mzs)
            & np.isfinite(intensities)
            & (mzs >= 0)
            & (intensities >= 0)
        )

        mzs = mzs[valid]
        intensities = intensities[valid]

        if len(mzs) == 0:
            return None

        # Keep the most intense peaks
        if len(mzs) > self.max_peaks:

            indices = np.argsort(intensities)[-self.max_peaks:]

            mzs = mzs[indices]
            intensities = intensities[indices]

        # Normalize fragment masses
        relative_mzs = mzs / precursor_mz

        peaks = np.stack(
            [relative_mzs, intensities],
            axis=1
        )

        precursor = np.array(
            [precursor_mz / 1000.0],
            dtype=np.float32
        )

        return (
            torch.from_numpy(peaks),
            torch.from_numpy(precursor),
            torch.from_numpy(fingerprint)
        )

    def __iter__(self):

        parquet_file = pq.ParquetFile(self.parquet_path)

        columns = [
            "normalized_smiles",
            "inchikey14",
            "precursor_mz",
            "ms2_mzs",
            "ms2_normalized_intensities"
        ]

        # Buffer used to shuffle training samples
        buffer = []

        rng = random.Random(42)

        for batch in parquet_file.iter_batches(
            batch_size=self.batch_size,
            columns=columns
        ):

            # Convert only this batch to Python lists
            data = {
                name: batch.column(name).to_pylist()
                for name in columns
            }

            for i in range(batch.num_rows):

                sample = self.process_row(
                    data["normalized_smiles"][i],
                    data["inchikey14"][i],
                    data["precursor_mz"][i],
                    data["ms2_mzs"][i],
                    data["ms2_normalized_intensities"][i]
                )

                if sample is None:
                    continue

                if self.shuffle_buffer_size == 0:

                    yield sample

                else:

                    buffer.append(sample)

                    if len(buffer) >= self.shuffle_buffer_size:

                        index = rng.randrange(len(buffer))

                        yield buffer.pop(index)

        # Empty remaining shuffle buffer
        rng.shuffle(buffer)

        yield from buffer