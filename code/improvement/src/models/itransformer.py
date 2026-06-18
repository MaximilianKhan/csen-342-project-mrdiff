"""iTransformer with trend/residual decomposition.

Inverts the attention axis: each variable's full lookback becomes one token.
Self-attention runs over D channel tokens (7x7 for multivariate ETT),
directly learning cross-variate dynamics that CI cannot.

For univariate (D=1), this degenerates to trivial 1-token attention.
Use CI transformer for univariate.

Reference: Liu et al., "iTransformer," ICLR 2024.
"""

import torch
import torch.nn as nn


class ITransformerDecomp(nn.Module):
    """lookback -> trend/residual decomp -> embed channels -> transformer
    over D tokens -> dual heads -> sum forecasts."""

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

        self.input_proj = nn.Linear(lookback_length, d_model)
        self.channel_embed = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head_norm = nn.LayerNorm(d_model)

        self.trend_head = nn.Linear(d_model, forecast_length)
        self.resid_head = nn.Linear(d_model, forecast_length)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def _project_component(self, component, head):
        """[B, D, H] -> embed -> transformer -> head -> [B, T, D]"""
        x = self.input_proj(component) + self.channel_embed
        x = self.transformer(x)
        x = self.head_norm(x)
        return head(x).transpose(1, 2)

    def forward(self, lookback):
        """[B, H, D] -> [B, T, D]"""
        x = lookback.transpose(1, 2)
        ks = self.trend_kernel
        pad = ks // 2
        x_padded = nn.functional.pad(x, (pad, pad), mode='replicate')
        trend = nn.functional.avg_pool1d(x_padded, ks, stride=1)
        resid = x - trend

        return (
            self._project_component(trend, self.trend_head)
            + self._project_component(resid, self.resid_head)
        )
