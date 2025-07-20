# ai_ecg/models.py

import torch
import torch.nn as nn

class TwoTower(nn.Module):
    """
    Two‐tower fusion network:
      • Waveform branch: 1‐D CNN over 8‐lead ECG (shape [B, 8, 2500])
      • Tabular branch: MLP over your tabular features (dim = tab_in_dim)
      • Fusion head: concatenates the two embeddings and predicts a single probability.
    """
    def __init__(self, tab_in_dim: int):
        super().__init__()
        # ─── Waveform branch ──────────────────────────────────────────────────
        self.cnn = nn.Sequential(
            # Input: [B, 8, 2500]
            nn.Conv1d(8,  32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),                         nn.Dropout(0.2),
            # [B,32,1250]
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),                            nn.Dropout(0.2),
            # [B,64,625]
            nn.Conv1d(64,128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),                             nn.Dropout(0.2),
            # [B,128,312]
            nn.AdaptiveAvgPool1d(1)   # → [B,128,1]
        )

        # ─── Tabular branch ───────────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(tab_in_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,          32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
        )

        # ─── Fusion & classifier ─────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,         16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16,          1), nn.Sigmoid()
        )

    def forward(self, ecg: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        """
        ecg: [batch, 8, 2500] tensor of ECG waveforms
        tab: [batch, tab_in_dim] tensor of tabular features
        returns: [batch] tensor of probabilities (0–1)
        """
        # waveform embedding
        x1 = self.cnn(ecg)          # [B,128,1]
        x1 = x1.view(x1.size(0), -1)  # [B,128]

        # tabular embedding
        x2 = self.mlp(tab)           # [B,32]

        # fusion & output
        x  = torch.cat([x1, x2], dim=1)  # [B,160]
        return self.classifier(x).squeeze(1)  # [B]

