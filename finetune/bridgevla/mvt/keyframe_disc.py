'''
Keyframe discriminator for BridgeVLA++ episodic memory.

Predicts, per step, whether the CURRENT frame is a "subtask-boundary"
keyframe that should be admitted into the episodic temporal memory
(label 1 == last keyframe of a subtask). Trained with BCE on the GT
``mem_label`` derived from keyframe language segments; at inference its
sigmoid output gates whether the frame is pushed into the MemoryBank.

Input (option A): the CURRENT frame's POST-memory stage-1 token grid
``x_heads`` of shape ``(bs*V, dim, Hp, Wp)`` (the same tensor the heatmap
head consumes). Because these tokens have already cross-attended the
existing memory keyframes, "did the current frame complete a new subtask
relative to what is already remembered" is implicit in the features.

Architecture: attention pooling (one learned query attends all V*Hp*Wp
tokens) -> LayerNorm -> MLP -> 1 logit.
Lives in fp32 (mirrors the memory blocks); the forward casts inputs to the
parameter dtype so a bf16 backbone does not break it.

Author: (generated for the variable-length keyframe-gated memory redesign)
'''
import torch
from torch import nn


class KeyframeDiscriminator(nn.Module):
    def __init__(
        self,
        dim: int = 2048,
        num_views: int = 3,
        heads: int = 8,
        hidden: int = 512,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_views = int(num_views)
        # Single learned query that attention-pools the token grid.
        self.query = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            self.dim, num_heads=int(heads), batch_first=True,
        )
        self.norm = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x_heads, num_views: int = None):
        """x_heads: (bs*V, dim, Hp, Wp) post-memory stage-1 tokens.

        Returns a (bs,) logit (NOT sigmoid'd) for BCEWithLogits.
        """
        V = int(num_views) if num_views is not None else self.num_views
        bsV, dim, Hp, Wp = x_heads.shape
        assert dim == self.dim, (dim, self.dim)
        assert bsV % V == 0, (bsV, V)
        bs = bsV // V
        # Match the discriminator's parameter dtype (fp32 by construction).
        x = x_heads.to(self.query.dtype)
        # (bs*V, dim, Hp, Wp) -> (bs, V, N, dim)
        N = Hp * Wp
        x = x.reshape(bs, V, dim, N).permute(0, 1, 3, 2)        # (bs, V, N, dim)
        x = x.reshape(bs, V * N, dim)                            # (bs, V*N, dim)
        q = self.query.expand(bs, 1, dim)                       # (bs, 1, dim)
        pooled, _ = self.attn(q, x, x, need_weights=False)      # (bs, 1, dim)
        pooled = self.norm(pooled.squeeze(1))                    # (bs, dim)
        logit = self.mlp(pooled).squeeze(-1)                    # (bs,)
        return logit
