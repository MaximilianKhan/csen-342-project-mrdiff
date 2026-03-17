"""CI+Decomp Transformer with Attention Residuals (Exp 26).

Replaces standard residual connections in the transformer layers with
AttnRes: each layer can selectively attend over ALL prior layer outputs
via learned pseudo-queries, not just its immediate predecessor.

Combined with data augmentation during training for more diverse signal.
"""

import math
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm for AttnRes keys."""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms


class AttnResTransformerLayer(nn.Module):
    """Transformer layer with AttnRes instead of standard residual connections.

    Standard transformer: x = x + self_attn(norm(x)); x = x + ffn(norm(x))
    AttnRes transformer: x = attnres(layers) + self_attn(norm(attnres(layers)));
                          output stored for future layers to attend over.

    The AttnRes pseudo-query lets this layer selectively retrieve from any
    prior layer — not just the immediate predecessor.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        # Standard transformer components
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # AttnRes: pseudo-query for selective depth-wise retrieval
        # Zero-init → uniform attention at start = equivalent to standard residual
        self.attn_query = nn.Parameter(torch.zeros(d_model))
        self.key_norm = RMSNorm(d_model)

    def _attnres_aggregate(self, layer_outputs):
        """Aggregate prior layer outputs via AttnRes attention.

        Args:
            layer_outputs: list of [B, N, d] tensors from prior layers

        Returns:
            [B, N, d] — weighted combination of all prior outputs
        """
        # Stack: [num_sources, B, N, d]
        sources = torch.stack(layer_outputs, dim=0)
        S, B, N, d = sources.shape

        # Keys: RMSNorm over d dimension
        keys = self.key_norm(sources)  # [S, B, N, d]

        # Attention logits: query · key, summed over d → [S, B, N]
        logits = torch.einsum('d, s b n d -> s b n', self.attn_query, keys)

        # Softmax over sources (depth dimension)
        attn = torch.softmax(logits, dim=0)  # [S, B, N]

        # Weighted sum
        return (attn.unsqueeze(-1) * sources).sum(dim=0)  # [B, N, d]

    def forward(self, layer_outputs):
        """
        Args:
            layer_outputs: list of prior layer outputs (including embedding)

        Returns:
            new output [B, N, d] to append to layer_outputs
        """
        # AttnRes: selectively aggregate all prior layers
        x = self._attnres_aggregate(layer_outputs)

        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)

        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))

        return x


class CIAttnResDecompTransformer(nn.Module):
    """Channel-Independent Decomposed Transformer with Attention Residuals.

    Combines:
    - CI patching (channel independence, shared weights)
    - Trend/residual decomposition (DLinear prior)
    - AttnRes connections (selective depth-wise retrieval)
    """

    def __init__(
        self,
        input_dim, forecast_length, lookback_length,
        patch_size=16, d_model=64, nhead=4, num_layers=3,
        dim_feedforward=128, dropout=0.3, trend_kernel=25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.forecast_length = forecast_length
        self.lookback_length = lookback_length
        self.patch_size = patch_size
        self.n_patches = lookback_length // patch_size
        self.d_model = d_model
        self.trend_kernel = trend_kernel

        # Shared patch embedding
        self.patch_embed = nn.Linear(patch_size, d_model)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # AttnRes transformer layers (replace standard TransformerEncoder)
        self.layers = nn.ModuleList([
            AttnResTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Separate heads for trend and residual
        self.head_norm = nn.LayerNorm(d_model)

        self.trend_temporal = nn.Linear(self.n_patches, forecast_length)
        self.trend_channel = nn.Linear(d_model, 1)

        self.resid_temporal = nn.Linear(self.n_patches, forecast_length)
        self.resid_channel = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)
        # Re-zero AttnRes queries after init
        for layer in self.layers:
            nn.init.zeros_(layer.attn_query)

    def _encode_and_project(self, x_1d, temporal_proj, channel_proj, B, D):
        """Shared encode + project for one component."""
        x = x_1d[:, :self.n_patches * self.patch_size]
        x = x.reshape(-1, self.n_patches, self.patch_size)

        # Patch embed + position
        x = self.patch_embed(x) + self.pos_embed  # [B*D, N, d]

        # AttnRes transformer: collect all layer outputs
        layer_outputs = [x]  # v_0 = embedding
        for layer in self.layers:
            v_l = layer(layer_outputs)
            layer_outputs.append(v_l)

        # Use final layer output
        x = self.head_norm(layer_outputs[-1])

        # CI head projections
        x = temporal_proj(x.transpose(1, 2))  # [B*D, d, T]
        x = channel_proj(x.transpose(1, 2))   # [B*D, T, 1]
        return x.squeeze(-1).reshape(B, D, self.forecast_length).transpose(1, 2)

    def forward(self, lookback):
        """lookback: [B, H, D] -> [B, T, D]"""
        B, H, D = lookback.shape

        # Trend/residual decomposition
        x = lookback.transpose(1, 2)
        ks = self.trend_kernel
        pad = ks // 2
        x_padded = nn.functional.pad(x, (pad, pad), mode='replicate')
        trend = nn.functional.avg_pool1d(x_padded, ks, stride=1)
        resid = x - trend

        # CI flatten
        trend_flat = trend.reshape(B * D, H)
        resid_flat = resid.reshape(B * D, H)

        # Process each through shared AttnRes transformer
        trend_out = self._encode_and_project(
            trend_flat, self.trend_temporal, self.trend_channel, B, D)
        resid_out = self._encode_and_project(
            resid_flat, self.resid_temporal, self.resid_channel, B, D)

        return trend_out + resid_out
