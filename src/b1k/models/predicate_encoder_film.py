"""Deep Sets predicate encoder with FiLM modulation (v3_film).

Fixes the state signal drowning problem in the original Deep Sets encoder (v2_deep_sets):
- Original: concat([identity_320, progress_1, state_1]) → phi MLP → state is 0.6% of input
- FiLM: state generates gamma/beta that MODULATE identity → state controls 100% of features

Also adds Fourier encoding of progress to fix MLP spectral bias for continuous values.

Reference:
- FiLM: Perez et al., "Visual Reasoning with a General Conditioning Layer", AAAI 2018
- Fourier features: Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions", NeurIPS 2020
"""

from typing import Dict, List

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from b1k.models.pi_behavior_config import MAX_NUM_PREDICATES, TASK_NUM_PREDICATES

# Denominators M per predicate per task (from V2 smoothed pkl).
# progress = k/M where k is integer. Used to snap continuous self-predicted
# progress to the discrete grid the model was trained on.
# Verified exhaustively: all 20 V2 tasks × all predicates match pkl data.
_PROGRESS_M: Dict[int, List[int]] = {
    2: [2,3,1,1], 3: [2,2,2,1], 5: [4,2], 6: [3,3],
    10: [1,1,1,1,1,1], 11: [8,6], 13: [1,1,1], 14: [1,1,1],
    15: [3], 19: [1,1,1,1,1,1,1], 23: [6], 24: [2,2,2,2],
    25: [1,1,2,1], 28: [1,1,1,1,1,1,1,1,1,1], 29: [2,2,2,1,1,1,1,1],
    34: [1], 42: [1,1,1,1], 44: [1,1,1,1,1,1,1,1],
    47: [2,2,1,1], 48: [1,1,1,1,2,1,1],
}


# Pre-built M lookup table: [num_tasks, MAX_NUM_PREDICATES]
# Constant, built once at import time, not during JIT trace.
import numpy as _np
_M_TABLE_NP = _np.ones((len(TASK_NUM_PREDICATES), MAX_NUM_PREDICATES), dtype=_np.float32)
for _tid, _ms in _PROGRESS_M.items():
    for _i, _m in enumerate(_ms):
        _M_TABLE_NP[_tid, _i] = float(_m)
_M_TABLE = jnp.array(_M_TABLE_NP)
del _M_TABLE_NP  # cleanup


def _snap_progress_batch(progress, task_ids):
    """Snap progress values to nearest discrete grid point (k/M) per predicate.

    Pure JAX ops — fully JIT compatible.
    For task_ids not in _PROGRESS_M (M=1 default), snap is round(x) = 0 or 1.
    During training with GT progress, this is a no-op (GT is already discrete).
    During eval, snaps continuous self-pred/oracle values to training distribution.

    Args:
        progress: [B, P] continuous progress values
        task_ids: [B] task indices

    Returns:
        [B, P] snapped progress values
    """
    m_vals = _M_TABLE[task_ids]  # [B, P] — constant lookup, no trace-time loop
    snapped = jnp.round(progress * m_vals) / m_vals
    return jnp.clip(snapped, 0.0, 1.0)


def fourier_encode(x, num_freqs: int = 8):
    """Encode scalar x ∈ [0,1] using sinusoidal positional encoding.

    Maps 1-dim scalar to 2*num_freqs dims using sin/cos at exponentially
    increasing frequencies. Lets the MLP distinguish nearby values
    (e.g., 0.33 vs 0.50) that would otherwise be indistinguishable.

    Args:
        x: [...] scalar values in [0, 1]
        num_freqs: number of frequency bands

    Returns:
        [..., 2*num_freqs] encoded features
    """
    freqs = 2.0 ** jnp.arange(num_freqs) * jnp.pi  # [num_freqs]
    # Broadcast: x[..., None] * freqs → [..., num_freqs]
    angles = x[..., None] * freqs
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)  # [..., 2*num_freqs]


class PredicateEncoderFiLM(nnx.Module):
    """Deep Sets encoder with FiLM state modulation.

    Architecture:
        1. Identity features: type_emb(64) + name_emb(128) + arg_emb(128) = 320 dims
        2. State features: Fourier(progress, 8 freqs) + satisfied = 17 dims → MLP → 64 dims
        3. FiLM: gamma(64→320) * identity + beta(64→320) = 320 dims (state controls every channel)
        4. phi MLP: 320 → 512 → 1024
        5. Sum aggregation + count → rho MLP → 2048 → 4 tokens
    """

    def __init__(self, rngs: nnx.Rngs, predicate_encoding_dim: int = 1024,
                 task_embedding_dim: int = 2048, num_fourier_freqs: int = 8):
        self.num_fourier_freqs = num_fourier_freqs
        fourier_dim = 2 * num_fourier_freqs  # 16
        state_input_dim = fourier_dim + 1  # 16 (fourier progress) + 1 (satisfied) = 17
        state_hidden_dim = 64
        identity_dim = 64 + 128 + 128  # type + name + arg = 320

        # Shared predicate feature embeddings (same as v2_deep_sets)
        self.pred_type_emb = nnx.Embed(num_embeddings=6, features=64, rngs=rngs)
        self.pred_name_emb = nnx.Embed(num_embeddings=20, features=128, rngs=rngs)
        self.pred_arg_emb = nnx.Embed(num_embeddings=256, features=128, rngs=rngs)

        # State encoder: [fourier_progress(16), satisfied(1)] → 64
        self.state_fc1 = nnx.Linear(state_input_dim, state_hidden_dim, rngs=rngs)
        self.state_fc2 = nnx.Linear(state_hidden_dim, state_hidden_dim, rngs=rngs)

        # FiLM generators: state_encoded(64) → gamma(320) and beta(320)
        # Initialize gamma bias=1 (identity scale), beta bias=0 (no shift)
        # so untrained FiLM = identity transform
        self.film_gamma = nnx.Linear(state_hidden_dim, identity_dim, rngs=rngs)
        self.film_beta = nnx.Linear(state_hidden_dim, identity_dim, rngs=rngs)

        # phi MLP: modulated_identity(320) → 512 → 1024
        self.phi_fc1 = nnx.Linear(identity_dim, 512, rngs=rngs)
        self.phi_fc2 = nnx.Linear(512, predicate_encoding_dim, rngs=rngs)

        # rho MLP: [summed(1024) + count(1)] → 1024 → 2048
        self.rho_fc1 = nnx.Linear(predicate_encoding_dim + 1, predicate_encoding_dim, rngs=rngs)
        self.rho_fc2 = nnx.Linear(predicate_encoding_dim, task_embedding_dim, rngs=rngs)

        # Project to 4 tokens
        self.pred_token_proj = nnx.Linear(task_embedding_dim, task_embedding_dim * 4, rngs=rngs)

        # Zero-init FiLM generators (AdaLN-Zero style):
        # Weights=0 + bias=1 for gamma → output is all 1s (identity scale)
        # Weights=0 + bias=0 for beta → output is all 0s (no shift)
        # This ensures FiLM starts as exact identity transform before training
        self.film_gamma.kernel.value = jnp.zeros_like(self.film_gamma.kernel.value)
        self.film_gamma.bias.value = jnp.ones_like(self.film_gamma.bias.value)
        self.film_beta.kernel.value = jnp.zeros_like(self.film_beta.kernel.value)
        self.film_beta.bias.value = jnp.zeros_like(self.film_beta.bias.value)

    def __call__(self, obs, task_ids):
        """Encode predicates with FiLM modulation.

        Args:
            obs: Observation with predicate_states, predicate_mask, predicate_progress,
                 predicate_type_ids, predicate_name_ids, predicate_arg_ids
            task_ids: [B] task indices (for progress clamping to discrete grid)

        Returns:
            [B, 4, 2048] predicate-conditioned tokens
        """
        B = obs.predicate_states.shape[0]

        type_ids = getattr(obs, 'predicate_type_ids', None)
        if type_ids is None:
            type_ids = jnp.zeros((B, MAX_NUM_PREDICATES), dtype=jnp.int32)
        name_ids = getattr(obs, 'predicate_name_ids', None)
        if name_ids is None:
            name_ids = jnp.zeros((B, MAX_NUM_PREDICATES), dtype=jnp.int32)
        arg_ids = getattr(obs, 'predicate_arg_ids', None)
        if arg_ids is None:
            arg_ids = jnp.zeros((B, MAX_NUM_PREDICATES), dtype=jnp.int32)
        progress = getattr(obs, 'predicate_progress', None)
        if progress is None:
            progress = obs.predicate_states.astype(jnp.float32)

        # === Step 0: Snap progress to discrete grid ===
        # Guarantees encoder always sees in-distribution values (k/M).
        # Training: no-op (GT progress is already discrete).
        # Self-pred eval: snaps continuous sigmoid output to training grid.
        # Oracle eval: snaps jittery simulator values to clean discrete values.
        progress = _snap_progress_batch(progress, task_ids)

        # === Step 1: Identity features (static, per-predicate) ===
        type_feat = self.pred_type_emb(type_ids)     # [B, P, 64]
        name_feat = self.pred_name_emb(name_ids)     # [B, P, 128]
        arg_feat = self.pred_arg_emb(arg_ids)         # [B, P, 128]
        identity = jnp.concatenate([type_feat, name_feat, arg_feat], axis=-1)  # [B, P, 320]

        # === Step 2: State features (dynamic) ===
        progress_fourier = fourier_encode(progress, self.num_fourier_freqs)  # [B, P, 16]
        satisfied = obs.predicate_states.astype(jnp.float32)[..., None]       # [B, P, 1]
        state_input = jnp.concatenate([progress_fourier, satisfied], axis=-1)  # [B, P, 17]
        state_encoded = nnx.relu(self.state_fc1(state_input))                  # [B, P, 64]
        state_encoded = nnx.relu(self.state_fc2(state_encoded))                # [B, P, 64]

        # === Step 3: FiLM modulation ===
        gamma = self.film_gamma(state_encoded)  # [B, P, 320] — scale (init ~1)
        beta = self.film_beta(state_encoded)    # [B, P, 320] — shift (init ~0)
        modulated = gamma * identity + beta      # [B, P, 320] — state controls every channel

        # === Step 4: phi MLP ===
        h = nnx.relu(self.phi_fc1(modulated))   # [B, P, 512]
        phi_out = self.phi_fc2(h)                # [B, P, 1024]

        # === Step 5: Masked sum aggregation ===
        mask = obs.predicate_mask[..., None].astype(jnp.float32)  # [B, P, 1]
        summed = jnp.sum(phi_out * mask, axis=1)  # [B, 1024]
        num_valid = jnp.maximum(jnp.sum(mask, axis=1), 1.0)  # [B, 1]

        # === Step 6: rho MLP ===
        rho_input = jnp.concatenate([summed, num_valid], axis=-1)  # [B, 1025]
        rho_h = nnx.relu(self.rho_fc1(rho_input))   # [B, 1024]
        predicate_repr = self.rho_fc2(rho_h)          # [B, 2048]

        # === Step 7: Project to 4 tokens ===
        tokens_flat = self.pred_token_proj(predicate_repr)  # [B, 4*2048]
        pred_tokens = tokens_flat.reshape(B, 4, -1)          # [B, 4, 2048]

        return pred_tokens
