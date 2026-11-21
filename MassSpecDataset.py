import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

class MassSpecDataset(Dataset):

    def __init__(self, df, fingerprint_dict, max_peaks=200):

        self.df = df.reset_index(drop=True)
        self.fingerprint_dict = fingerprint_dict
        self.max_peaks = max_peaks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        mzs = np.asarray(row["ms2_mzs"], dtype=np.float32)
        intensities = np.asarray(
            row["ms2_normalized_intensities"],
            dtype=np.float32
        )

        # Keep only valid corresponding peaks
        n = min(len(mzs), len(intensities))

        mzs = mzs[:n]
        intensities = intensities[:n]

        valid = (
            np.isfinite(mzs)
            & np.isfinite(intensities)
            & (mzs >= 0)
        )

        mzs = mzs[valid]
        intensities = intensities[valid]

        # Retain the most intense peaks
        if len(mzs) > self.max_peaks:

            indices = np.argsort(intensities)[-self.max_peaks:]

            mzs = mzs[indices]
            intensities = intensities[indices]

        precursor_mz = float(row["precursor_mz"])

        # Normalize fragment masses
        relative_mzs = mzs / precursor_mz

        # Each peak has two features
        peaks = np.stack(
            [relative_mzs, intensities],
            axis=1
        )

        # Normalize precursor mass
        precursor = np.array(
            [precursor_mz / 1000.0],
            dtype=np.float32
        )

        # Target fingerprint
        fingerprint = self.fingerprint_dict[
            row["normalized_smiles"]
        ]

        return (
            torch.tensor(peaks, dtype=torch.float32),
            torch.tensor(precursor, dtype=torch.float32),
            torch.tensor(fingerprint, dtype=torch.float32)
        )



