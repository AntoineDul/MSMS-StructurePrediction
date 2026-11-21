import torch.nn as nn
import torch

class DeepSets(nn.Module):

    def __init__(self, peak_dim=2, hidden_dim=128, fingerprint_size=1024, pooling="sum"):

        super().__init__()

        self.pooling = pooling

        # phi -> encode each individual peak
        self.phi = nn.Sequential(
            nn.Linear(peak_dim, 64),
            nn.ReLU(),

            nn.Linear(64, hidden_dim),
            nn.ReLU()
        )

        # rho -> predict fingerprint from aggregated peaks
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim + 1, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, fingerprint_size)
        )

    def forward(self, peaks, mask, precursor):

        # peaks: (B, N, 2)
        # mask: (B, N)

        # Encode each peak independently
        embeddings = self.phi(peaks)

        # embeddings: (B, N, hidden_dim)

        # Remove padded peaks from aggregation
        embeddings = embeddings * mask.unsqueeze(-1)

        # Aggregate peak embeddings
        if self.pooling == "sum":

            pooled = embeddings.sum(dim=1)

        elif self.pooling == "mean":

            count = mask.sum(dim=1, keepdim=True).clamp(min=1)

            pooled = embeddings.sum(dim=1) / count

        else:
            raise ValueError("Unknown pooling method")

        # Add precursor information
        representation = torch.cat(
            [pooled, precursor],
            dim=1
        )

        # Predict fingerprint logits
        logits = self.rho(representation)

        return logits
