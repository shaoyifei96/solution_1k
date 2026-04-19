"""V1 predicate encoder with soft pooling (v2_soft).

Fixes V1's hard binary partition with progress-weighted soft pooling:
- Original V1: done_mask = (state == True), remaining_mask = (state == False)
  → hard partition, can't handle continuous progress (0.33, 0.67)
- Soft pooling: done_pool = Σ(emb × progress) / Σ(progress)
  → degenerates to V1 for binary (progress=0 or 1), smooth for continuous

Also adds Fourier encoding of progress for the soft pooling weights.

Uses V1's task-specific predicate embeddings (not shared Deep Sets).
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


class PredicateEncoderSoft:
    """V1 encoder with soft pooling.

    Architecture:
        1. Task-specific predicate embeddings (1024-dim each, from V1 embedding table)
        2. Soft pooling: progress-weighted done/remaining pools
           done_pool = Σ(emb_i × progress_i) / Σ(progress_i)
           remaining_pool = Σ(emb_i × (1 - progress_i)) / Σ(1 - progress_i)
        3. Gated fusion with task embedding (same as V1)
        4. Output: [B, 4, 2048]

    For binary predicates (progress ∈ {0, 1}), this is IDENTICAL to V1.
    For continuous predicates (progress ∈ [0, 1]), soft pooling smoothly distributes
    each embedding between done and remaining pools.
    """

    def __init__(self, rngs: nnx.Rngs, predicate_encoding_dim: int = 1024,
                 task_embedding_dim: int = 2048):
        # V1's task-specific predicate embedding table
        self.task_predicate_embeddings = nnx.Embed(
            num_embeddings=TOTAL_TASK_PREDICATE_EMBEDDINGS,
            features=predicate_encoding_dim,
            rngs=rngs,
        )

        # V1's gated fusion layers
        fusion_input_dim = task_embedding_dim + 2 * predicate_encoding_dim  # 2048 + 2*1024 = 4096
        self.gate_done = nnx.Linear(fusion_input_dim, predicate_encoding_dim, rngs=rngs)
        self.gate_remaining = nnx.Linear(fusion_input_dim, predicate_encoding_dim, rngs=rngs)
        self.gate_task = nnx.Linear(fusion_input_dim, task_embedding_dim, rngs=rngs)
        self.fusion_layer1 = nnx.Linear(fusion_input_dim, task_embedding_dim * 2, rngs=rngs)
        self.fusion_layer2 = nnx.Linear(task_embedding_dim * 2, task_embedding_dim, rngs=rngs)
        self.predicate_projection = nnx.Linear(2 * predicate_encoding_dim, task_embedding_dim, rngs=rngs)

    def aggregate_soft(self, task_ids, predicate_mask, predicate_progress):
        """Soft pooling: progress-weighted done/remaining aggregation.

        Args:
            task_ids: [B] task indices
            predicate_mask: [B, P] valid predicate mask
            predicate_progress: [B, P] progress values in [0, 1]

        Returns:
            done_agg: [B, 1024] — progress-weighted average (what's done)
            remaining_agg: [B, 1024] — (1-progress)-weighted average (what remains)
        """
        # Look up task-specific embeddings (same as V1)
        task_pred_offsets = jnp.array(TASK_PREDICATE_OFFSETS, dtype=jnp.int32)
        offsets = task_pred_offsets[task_ids]  # [B]
        pred_range = jnp.arange(MAX_NUM_PREDICATES)
        pred_indices = offsets[:, None] + pred_range[None, :]  # [B, P]
        max_idx = TOTAL_TASK_PREDICATE_EMBEDDINGS - 1
        pred_indices = jnp.clip(pred_indices, 0, max_idx)

        all_embeddings = self.task_predicate_embeddings(pred_indices)  # [B, P, 1024]

        # Soft pooling weights
        mask = predicate_mask.astype(jnp.float32)  # [B, P]
        progress = predicate_progress.astype(jnp.float32)  # [B, P]

        # Done weights = progress * mask
        done_weights = progress * mask  # [B, P]
        done_weights_sum = jnp.maximum(jnp.sum(done_weights, axis=1, keepdims=True), 1e-6)  # [B, 1]
        done_agg = jnp.sum(all_embeddings * done_weights[..., None], axis=1) / done_weights_sum  # [B, 1024]

        # Remaining weights = (1 - progress) * mask
        remaining_weights = (1.0 - progress) * mask  # [B, P]
        remaining_weights_sum = jnp.maximum(jnp.sum(remaining_weights, axis=1, keepdims=True), 1e-6)  # [B, 1]
        remaining_agg = jnp.sum(all_embeddings * remaining_weights[..., None], axis=1) / remaining_weights_sum  # [B, 1024]

        return done_agg, remaining_agg

    def __call__(self, task_embedding, task_ids, predicate_states, predicate_mask,
                 predicate_progress=None):
        """Encode predicates with soft pooling + gated fusion.

        Args:
            task_embedding: [B, 2048] base task embedding
            task_ids: [B] task indices
            predicate_states: [B, P] binary predicate states
            predicate_mask: [B, P] valid predicate mask
            predicate_progress: [B, P] continuous progress (if None, uses binary states)

        Returns:
            [B, 4, 2048] fused task+predicate tokens
        """
        # Use binary states as progress if no continuous progress available
        if predicate_progress is None:
            predicate_progress = predicate_states.astype(jnp.float32)

        # Soft pooling
        done_agg, remaining_agg = self.aggregate_soft(
            task_ids, predicate_mask, predicate_progress
        )  # [B, 1024], [B, 1024]

        # Gated fusion (identical to V1's fuse_task_and_predicates)
        all_inputs = jnp.concatenate([
            task_embedding,   # [B, 2048]
            done_agg,         # [B, 1024]
            remaining_agg     # [B, 1024]
        ], axis=-1)  # [B, 4096]

        gate_done = nnx.sigmoid(self.gate_done(all_inputs))            # [B, 1024]
        gate_remaining = nnx.sigmoid(self.gate_remaining(all_inputs))  # [B, 1024]
        gate_task = nnx.sigmoid(self.gate_task(all_inputs))            # [B, 2048]

        # 1. Task-gated representation
        task_gated = task_embedding * gate_task  # [B, 2048]

        # 2. Balanced fusion
        x = nnx.relu(self.fusion_layer1(all_inputs))  # [B, 4096]
        balanced_fusion = self.fusion_layer2(x)         # [B, 2048]

        # 3. Remaining-focus representation
        gated_remaining = jnp.concatenate([
            done_agg * gate_done,
            remaining_agg * gate_remaining
        ], axis=-1)  # [B, 2048]
        gated_predicate_proj = self.predicate_projection(gated_remaining)  # [B, 2048]

        # 4. Done-focus representation
        raw_predicates = jnp.concatenate([done_agg, remaining_agg], axis=-1)  # [B, 2048]

        # Stack all four representations
        fused_embeddings = jnp.stack([
            task_gated, balanced_fusion, gated_predicate_proj, raw_predicates
        ], axis=1)  # [B, 4, 2048]

        return fused_embeddings
