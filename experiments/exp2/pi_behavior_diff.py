"""
Exp 2: Changes to src/b1k/models/pi_behavior.py

3 changes:
1. __init__: Add forall_module, exists_module, progress_pred_from_vlm
2. aggregate_predicate_embeddings: Apply per-type modules with progress
3. Loss: Add MSE progress loss

--- CHANGE 1: Add after line 177 (after self.predicate_projection) ---
"""

# Add these lines after self.predicate_projection in __init__:
INIT_ADDITIONS = """
        # V2: Per-type predicate modules (progress-aware)
        # ForallModule: MLP([embedding; progress_ratio]) + residual
        self.forall_fc1 = nnx.Linear(self.predicate_encoding_dim + 1, self.predicate_encoding_dim, rngs=rngs)
        self.forall_fc2 = nnx.Linear(self.predicate_encoding_dim, self.predicate_encoding_dim, rngs=rngs)
        # ExistsModule: same structure
        self.exists_fc1 = nnx.Linear(self.predicate_encoding_dim + 1, self.predicate_encoding_dim, rngs=rngs)
        self.exists_fc2 = nnx.Linear(self.predicate_encoding_dim, self.predicate_encoding_dim, rngs=rngs)
        # Progress prediction head (MSE loss)
        self.progress_pred_from_vlm = nnx.Linear(paligemma_config.width, MAX_NUM_PREDICATES, rngs=rngs)
"""


# --- CHANGE 2: Replace aggregate_predicate_embeddings (lines 422-475) ---

AGGREGATE_REPLACEMENT = """
    def aggregate_predicate_embeddings(
        self,
        task_ids,
        predicate_states,
        predicate_mask,
        predicate_progress=None,
    ):
        \"\"\"Aggregate predicate embeddings with per-type progress modules.

        Args:
            task_ids: [B]
            predicate_states: [B, P] bool
            predicate_mask: [B, P] bool
            predicate_progress: [B, P] float (0-1 continuous), or None

        Returns:
            done_agg: [B, 1024]
            remaining_agg: [B, 1024]
        \"\"\"
        batch_size = task_ids.shape[0]
        task_pred_offsets = jnp.array(TASK_PREDICATE_OFFSETS, dtype=jnp.int32)
        task_num_preds = jnp.array(TASK_NUM_PREDICATES, dtype=jnp.int32)

        offsets = task_pred_offsets[task_ids]
        num_preds = task_num_preds[task_ids]

        pred_range = jnp.arange(MAX_NUM_PREDICATES)
        pred_indices = offsets[:, None] + pred_range[None, :]
        max_idx = TOTAL_TASK_PREDICATE_EMBEDDINGS - 1
        pred_indices = jnp.clip(pred_indices, 0, max_idx)

        all_embeddings = self.task_predicate_embeddings(pred_indices)  # [B, P, 1024]

        # Apply per-type modules if progress is available
        if predicate_progress is not None:
            progress = predicate_progress[..., None]  # [B, P, 1]
            emb_with_progress = jnp.concatenate([all_embeddings, progress], axis=-1)  # [B, P, 1025]

            # Forall module: MLP + residual
            forall_out = nnx.relu(self.forall_fc1(emb_with_progress))
            forall_out = self.forall_fc2(forall_out) + all_embeddings  # residual

            # Exists module: MLP + residual
            exists_out = nnx.relu(self.exists_fc1(emb_with_progress))
            exists_out = self.exists_fc2(exists_out) + all_embeddings  # residual

            # Determine predicate types from progress values:
            # binary predicates have progress in {0.0, 1.0}, forall/exists have intermediate values
            # Simple heuristic: if progress != states (as float), it's a quantified predicate
            states_float = predicate_states.astype(jnp.float32)
            is_quantified = jnp.abs(predicate_progress - states_float) > 0.01  # [B, P]

            # Use forall_out for quantified predicates, original for binary
            # (We don't distinguish forall vs exists here — both use progress similarly)
            all_embeddings = jnp.where(
                is_quantified[..., None],
                forall_out,
                all_embeddings
            )

        done_mask = predicate_states & predicate_mask
        remaining_mask = ~predicate_states & predicate_mask

        done_sum = jnp.sum(all_embeddings * done_mask[..., None], axis=1)
        num_done = jnp.maximum(jnp.sum(done_mask, axis=1, keepdims=True), 1.0)
        done_agg = done_sum / num_done

        remaining_sum = jnp.sum(all_embeddings * remaining_mask[..., None], axis=1)
        num_remaining = jnp.maximum(jnp.sum(remaining_mask, axis=1, keepdims=True), 1.0)
        remaining_agg = remaining_sum / num_remaining

        return done_agg, remaining_agg
"""


# --- CHANGE 3: In fuse_task_and_predicates, pass predicate_progress ---
# Line 509: add predicate_progress parameter and pass to aggregate
FUSE_DIFF = """
# In fuse_task_and_predicates signature, add predicate_progress parameter:
#   def fuse_task_and_predicates(self, task_embedding, task_ids,
#       predicate_states, predicate_mask, predicate_progress=None):
#
# Line 509-511: pass predicate_progress:
#   done_agg, remaining_agg = self.aggregate_predicate_embeddings(
#       task_ids, predicate_states, predicate_mask, predicate_progress
#   )
"""

# --- CHANGE 4: In embed_prefix, pass obs.predicate_progress ---
# Line 601-604: add predicate_progress argument
EMBED_PREFIX_DIFF = """
# In embed_prefix, line 601-604:
#   fused_task_embeddings = self.fuse_task_and_predicates(
#       base_task_embedding, task_ids,
#       obs.predicate_states, obs.predicate_mask,
#       obs.predicate_progress  # NEW
#   )
"""

# --- CHANGE 5: Add progress MSE loss after line 952 ---
LOSS_ADDITION = """
        # V2: Progress regression loss (MSE on continuous predicates)
        if train and observation.predicate_progress is not None:
            progress_pred = jax.nn.sigmoid(self.progress_pred_from_vlm(base_task_output))  # [B, 20]
            progress_target = observation.predicate_progress.astype(jnp.float32)  # [B, 20]
            progress_mse = jnp.square(progress_pred - progress_target) * pred_mask * valid_pred_mask.astype(jnp.float32)
            num_valid_progress = jnp.maximum(jnp.sum(pred_mask * valid_pred_mask.astype(jnp.float32), axis=-1), 1.0)
            per_sample_progress_loss = jnp.sum(progress_mse, axis=-1) / num_valid_progress
            losses["progress_loss"] = jnp.mean(per_sample_progress_loss)
            progress_loss_value = self.config.progress_loss_weight * losses["progress_loss"]
        else:
            progress_loss_value = 0.0

        # Update total loss to include progress
        # losses["total_loss"] = losses["action_loss"] + fast_loss_value + predicate_loss_value + progress_loss_value
"""
