"""V1 predicate encoder with FiLM state modulation (v2_soft).

Keeps V1's task-specific embeddings and done/remaining hard partition,
but adds FiLM modulation so progress controls each embedding dimension
BEFORE pooling. This way even 1-predicate tasks get progress-dependent
representations in the done/remaining pools.

Ablation role:
- v2_soft: task-specific emb + FiLM + hard partition → tests V1 embeddings
- v3_film: shared emb + FiLM + sum pooling → tests Deep Sets embeddings
- FiLM is constant across both → isolates embedding type and pooling method

Uses same Fourier encoding and clamp logic as v3_film.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from b1k.models.pi_behavior_config import (
    TASK_NUM_PREDICATES,
    TASK_PREDICATE_OFFSETS,
    TOTAL_TASK_PREDICATE_EMBEDDINGS,
    MAX_NUM_PREDICATES,
)
from b1k.models.predicate_encoder_film import fourier_encode, _snap_progress_batch


class PredicateEncoderSoft:
    """V1 encoder with FiLM modulation + hard partition.

    Architecture:
        1. Task-specific predicate embeddings (1024-dim, from V1 table)
        2. FiLM: progress+state → gamma/beta that modulate each embedding
        3. Hard partition: done_pool = mean(modulated_emb where state==True)
        4. Gated fusion with task embedding (same as V1)
        5. Output: [B, 4, 2048]

    For binary predicates: FiLM modulates by 0/1 state, partition = V1.
    For continuous progress: FiLM modulates by progress level,
    so the SAME embedding produces different representations at different progress.
    """

    def __init__(self, rngs: nnx.Rngs, predicate_encoding_dim: int = 1024,
                 task_embedding_dim: int = 2048, num_fourier_freqs: int = 8):
        self.num_fourier_freqs = num_fourier_freqs
        fourier_dim = 2 * num_fourier_freqs  # 16
        state_input_dim = fourier_dim + 1  # 16 (fourier progress) + 1 (satisfied) = 17
        state_hidden_dim = 64

        # V1's task-specific predicate embedding table
        self.task_predicate_embeddings = nnx.Embed(
            num_embeddings=TOTAL_TASK_PREDICATE_EMBEDDINGS,
            features=predicate_encoding_dim,
            rngs=rngs,
        )

        # FiLM: state → gamma/beta for each predicate embedding
        self.state_fc1 = nnx.Linear(state_input_dim, state_hidden_dim, rngs=rngs)
        self.state_fc2 = nnx.Linear(state_hidden_dim, state_hidden_dim, rngs=rngs)
        self.film_gamma = nnx.Linear(state_hidden_dim, predicate_encoding_dim, rngs=rngs)
        self.film_beta = nnx.Linear(state_hidden_dim, predicate_encoding_dim, rngs=rngs)

        # Zero-init FiLM (identity at start)
        self.film_gamma.kernel.value = jnp.zeros_like(self.film_gamma.kernel.value)
        self.film_gamma.bias.value = jnp.ones_like(self.film_gamma.bias.value)
        self.film_beta.kernel.value = jnp.zeros_like(self.film_beta.kernel.value)
        self.film_beta.bias.value = jnp.zeros_like(self.film_beta.bias.value)

        # V1's gated fusion layers
        fusion_input_dim = task_embedding_dim + 2 * predicate_encoding_dim  # 4096
        self.gate_done = nnx.Linear(fusion_input_dim, predicate_encoding_dim, rngs=rngs)
        self.gate_remaining = nnx.Linear(fusion_input_dim, predicate_encoding_dim, rngs=rngs)
        self.gate_task = nnx.Linear(fusion_input_dim, task_embedding_dim, rngs=rngs)
        self.fusion_layer1 = nnx.Linear(fusion_input_dim, task_embedding_dim * 2, rngs=rngs)
        self.fusion_layer2 = nnx.Linear(task_embedding_dim * 2, task_embedding_dim, rngs=rngs)
        self.predicate_projection = nnx.Linear(2 * predicate_encoding_dim, task_embedding_dim, rngs=rngs)

    def __call__(self, task_embedding, task_ids, predicate_states, predicate_mask,
                 predicate_progress=None):
        """Encode predicates with FiLM + V1 hard partition.

        Args:
            task_embedding: [B, 2048] base task embedding
            task_ids: [B] task indices
            predicate_states: [B, P] binary predicate states
            predicate_mask: [B, P] valid predicate mask
            predicate_progress: [B, P] continuous progress (if None, uses binary states)

        Returns:
            [B, 4, 2048] fused task+predicate tokens
        """
        if predicate_progress is None:
            predicate_progress = predicate_states.astype(jnp.float32)

        # === Step 1: Look up task-specific embeddings (V1 style) ===
        task_pred_offsets = jnp.array(TASK_PREDICATE_OFFSETS, dtype=jnp.int32)
        offsets = task_pred_offsets[task_ids]
        pred_range = jnp.arange(MAX_NUM_PREDICATES)
        pred_indices = offsets[:, None] + pred_range[None, :]
        max_idx = TOTAL_TASK_PREDICATE_EMBEDDINGS - 1
        pred_indices = jnp.clip(pred_indices, 0, max_idx)

        all_embeddings = self.task_predicate_embeddings(pred_indices)  # [B, P, 1024]

        # === Step 2: Clamp progress + Fourier encode + FiLM ===
        progress = _snap_progress_batch(predicate_progress, task_ids)  # snap to k/M grid
        progress_fourier = fourier_encode(progress, self.num_fourier_freqs)  # [B, P, 16]
        satisfied = predicate_states.astype(jnp.float32)[..., None]          # [B, P, 1]
        state_input = jnp.concatenate([progress_fourier, satisfied], axis=-1) # [B, P, 17]

        state_encoded = nnx.relu(self.state_fc1(state_input))   # [B, P, 64]
        state_encoded = nnx.relu(self.state_fc2(state_encoded)) # [B, P, 64]

        gamma = self.film_gamma(state_encoded)  # [B, P, 1024]
        beta = self.film_beta(state_encoded)    # [B, P, 1024]

        modulated = gamma * all_embeddings + beta  # [B, P, 1024] — progress-aware embeddings

        # === Step 3: V1-style hard partition (done / remaining pools) ===
        done_mask = predicate_states & predicate_mask       # [B, P]
        remaining_mask = ~predicate_states & predicate_mask  # [B, P]

        done_sum = jnp.sum(modulated * done_mask[..., None], axis=1)
        num_done = jnp.maximum(jnp.sum(done_mask, axis=1, keepdims=True), 1.0)
        done_agg = done_sum / num_done  # [B, 1024]

        remaining_sum = jnp.sum(modulated * remaining_mask[..., None], axis=1)
        num_remaining = jnp.maximum(jnp.sum(remaining_mask, axis=1, keepdims=True), 1.0)
        remaining_agg = remaining_sum / num_remaining  # [B, 1024]

        # === Step 4: Gated fusion (identical to V1) ===
        all_inputs = jnp.concatenate([
            task_embedding, done_agg, remaining_agg
        ], axis=-1)  # [B, 4096]

        gate_done = nnx.sigmoid(self.gate_done(all_inputs))
        gate_remaining = nnx.sigmoid(self.gate_remaining(all_inputs))
        gate_task = nnx.sigmoid(self.gate_task(all_inputs))

        task_gated = task_embedding * gate_task
        x = nnx.relu(self.fusion_layer1(all_inputs))
        balanced_fusion = self.fusion_layer2(x)

        gated_remaining = jnp.concatenate([
            done_agg * gate_done,
            remaining_agg * gate_remaining
        ], axis=-1)
        gated_predicate_proj = self.predicate_projection(gated_remaining)

        raw_predicates = jnp.concatenate([done_agg, remaining_agg], axis=-1)

        fused_embeddings = jnp.stack([
            task_gated, balanced_fusion, gated_predicate_proj, raw_predicates
        ], axis=1)  # [B, 4, 2048]

        return fused_embeddings
