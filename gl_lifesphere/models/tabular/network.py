"""The rung-0 encoder: a plain feed-forward network over the flattened design matrix.

Per encoder doc §4's ablation ladder, rung 0 is "Flattened/tabular + plain
neural network — no structure at all." The network plays the same role a GNN
encoder plays for model 3: it maps a Subject's representation to `z in R^d`,
trained end-to-end on the stratified Cox objective. Stage two freezes
`embed()` and passes `z` through the one shared decoder every model is scored
through (`gl_lifesphere.survival.decoder`); the training-time risk head is
not used by the primary result, but is kept (not discarded) because #3 §3
pre-declares scoring it directly as the "end-to-end" diagnostic — a by-product
of stage one, labelled "best-case", never reported without the two-stage
number beside it.
"""

from __future__ import annotations

import torch
from torch import nn


class TabularEncoder(nn.Module):
    """`x in R^{input_dim} -> z in R^d`, plus a `Linear(d, 1, bias=False)` head used
    to train the encoder and, separately, for the end-to-end diagnostic (#3 §3).

    No bias on the head — an intercept is unidentifiable under the Cox partial
    likelihood (decoder doc §4.2), so a trainable one would drift under weight
    decay alone and mean nothing when read.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (32,),
        d: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        layers.append(nn.Linear(width, d))
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1, bias=False)
        self.d = d

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """`z`, the representation handed to the shared decoder once frozen."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`r`, the scalar risk score used only to train the encoder (stage one)."""
        return self.head(self.embed(x)).squeeze(-1)
