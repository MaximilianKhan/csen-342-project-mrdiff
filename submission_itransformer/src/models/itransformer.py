"""iTransformer with trend/residual decomposition (Exp 29).

Inverts the attention axis: instead of treating time patches as tokens
(PatchTST / CI style), each variable's full lookback series becomes one token.
Self-attention is then over the D channel tokens, directly learning cross-variate
relationships without exploding parameter count.

For D=7, attention is 7×7 = 49 scores. For CI with overlapping patches on ETTh1,
attention was 83×83 = 6,889 scores — but those 83 tokens can't communicate across
channels at all. iTransformer fixes exactly what CI misses on multivariate.

For univariate (D=1), iTransformer degenerates to 1 token with a trivial no-op
attention. Use the CI transformer for univariate — it's wired in train scripts
to route by dataset mode.

Reference: Liu et al., "iTransformer: Inverted Transformers Are Effective for
Time Series Forecasting," ICLR 2024.
"""

import torch
import torch.nn as nn
from typing import Optional


class ITransformerDecomp(nn.Module):
    """iTransformer with trend/residual decomposition.

    Architecture:
        lookback [B, H, D]
          → trend/residual decomposition (avg-pool)
          → embed each channel's lookback series to d_model: [B, D, d_model]
          → add learnable channel positional embedding
          → TransformerEncoder over D channel tokens
          → output head: Linear(d_model → T) per token → [B, D, T]
          → reshape → [B, T, D]
          → sum trend + residual forecasts
    """

    def __init__(
        self,
        input_dim: int,
        forecast_length: int,
        lookback_length: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        trend_kernel: int = 25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.forecast_length = forecast_length
        self.lookback_length = lookback_length
        self.d_model = d_model
        self.trend_kernel = trend_kernel

        # Input projection: embed each channel's lookback series
        # Each token = one full channel series: Linear(H → d_model)
        self.input_proj = nn.Linear(lookback_length, d_model)

        # Learnable channel positional embedding [D, d_model]
        # Lets the model distinguish channel identity (e.g. OT vs HUFL vs HULL)
        self.channel_embed = nn.Parameter(
            torch.randn(1, input_dim, d_model) * 0.02
        )

        # Transformer over channel tokens (D tokens, not N_patch tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head_norm = nn.LayerNorm(d_model)

        # Separate output heads for trend and residual
        # Linear(d_model → T) maps each channel token to its forecast
        self.trend_head = nn.Linear(d_model, forecast_length)
        self.resid_head = nn.Linear(d_model, forecast_length)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def _project_component(self, component: torch.Tensor, head: nn.Linear) -> torch.Tensor:
        """Project one decomposition component through iTransformer.

        Args:
            component: [B, D, H] — one channel per row
            head: Linear(d_model → T)

        Returns:
            [B, T, D]
        """
        B, D, H = component.shape

        # Embed: [B, D, H] → [B, D, d_model]
        x = self.input_proj(component)

        # Add channel positional embedding
        x = x + self.channel_embed  # [B, D, d_model]

        # Transformer over D channel tokens: [B, D, d_model] → [B, D, d_model]
        x = self.transformer(x)
        x = self.head_norm(x)

        # Output head: [B, D, d_model] → [B, D, T] → [B, T, D]
        return head(x).transpose(1, 2)

    def forward(self, lookback: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lookback: [B, H, D]
        Returns:
            forecast: [B, T, D]
        """
        # [B, H, D] → [B, D, H] for channel-first processing
        x = lookback.transpose(1, 2)

        # Trend/residual decomposition
        ks = self.trend_kernel
        pad = ks // 2
        x_padded = nn.functional.pad(x, (pad, pad), mode='replicate')
        trend = nn.functional.avg_pool1d(x_padded, ks, stride=1)  # [B, D, H]
        resid = x - trend                                           # [B, D, H]

        # Project each component through iTransformer with its own head
        trend_forecast = self._project_component(trend, self.trend_head)  # [B, T, D]
        resid_forecast = self._project_component(resid, self.resid_head)  # [B, T, D]

        return trend_forecast + resid_forecast
