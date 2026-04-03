"""
Exp 3: Full V2 Deep Sets — changes to src/b1k/models/pi_behavior.py

Major changes:
1. __init__: REMOVE old predicate params, ADD Deep Sets encoder
2. REPLACE aggregate_predicate_embeddings + fuse_task_and_predicates with encode_predicates_deep_sets
3. MODIFY embed_prefix to use new encoder (still 5 tokens)
4. ADD progress MSE loss (same as Exp 2)

--- CHANGE 1: Replace predicate params in __init__ (lines 146-177) ---
"""

INIT_REMOVE = """
# DELETE these lines (146-177):
        self.predicate_pred_from_vlm = ...
        self.predicate_encoding_dim = ...
        self.task_predicate_embeddings = ...
        self.gate_done = ...
        self.gate_remaining = ...
        self.gate_task = ...
        self.fusion_layer1 = ...
        self.fusion_layer2 = ...
        self.predicate_projection = ...
"""

INIT_ADD = """
        # ====== V2 Deep Sets Predicate Encoder ======
        self.predicate_encoding_dim = config.task_embedding_dim // 2  # 1024

        # Predicate feature embeddings (shared across tasks)
        self.pred_type_emb = nnx.Embed(num_embeddings=6, features=64, rngs=rngs)     # atomic/forall/exists/not/forpairs/or
        self.pred_name_emb = nnx.Embed(num_embeddings=20, features=128, rngs=rngs)   # inside/ontop/nextto/...
        self.pred_arg_emb = nnx.Embed(num_embeddings=256, features=128, rngs=rngs)   # entity categories

        # phi: per-predicate MLP [64+128+128+1+1=322] -> hidden -> encoding_dim
        phi_input_dim = 64 + 128 + 128 + 1 + 1  # type + name + arg + progress + satisfied
        self.phi_fc1 = nnx.Linear(phi_input_dim, 512, rngs=rngs)
        self.phi_fc2 = nnx.Linear(512, self.predicate_encoding_dim, rngs=rngs)  # -> 1024

        # Per-type modules (progress-aware, same as Exp 2)
        self.forall_fc1 = nnx.Linear(self.predicate_encoding_dim + 1, self.predicate_encoding_dim, rngs=rngs)
        self.forall_fc2 = nnx.Linear(self.predicate_encoding_dim, self.predicate_encoding_dim, rngs=rngs)
        self.exists_fc1 = nnx.Linear(self.predicate_encoding_dim + 1, self.predicate_encoding_dim, rngs=rngs)
        self.exists_fc2 = nnx.Linear(self.predicate_encoding_dim, self.predicate_encoding_dim, rngs=rngs)

        # rho: post-aggregation MLP [1024] -> hidden -> task_embedding_dim
        self.rho_fc1 = nnx.Linear(self.predicate_encoding_dim, self.predicate_encoding_dim, rngs=rngs)
        self.rho_fc2 = nnx.Linear(self.predicate_encoding_dim, config.task_embedding_dim, rngs=rngs)  # -> 2048

        # Project Deep Sets output to 4 task tokens (to match V1's 5-token structure)
        self.pred_token_proj = nnx.Linear(config.task_embedding_dim, config.task_embedding_dim * 4, rngs=rngs)  # -> 4*2048

        # Prediction heads
        self.predicate_pred_from_vlm = nnx.Linear(paligemma_config.width, MAX_NUM_PREDICATES, rngs=rngs)
        self.progress_pred_from_vlm = nnx.Linear(paligemma_config.width, MAX_NUM_PREDICATES, rngs=rngs)
"""


# --- CHANGE 2: New method replacing aggregate + fuse ---
ENCODE_DEEP_SETS = """
    def encode_predicates_deep_sets(self, obs):
        \"\"\"V2 Deep Sets predicate encoder.

        Layer 1: Shared MLP phi per predicate
        Layer 2: Per-type modules (forall/exists with progress)
        Layer 3: Sum aggregation + MLP rho

        Returns: [B, 4, 2048] — 4 predicate-conditioned tokens
        \"\"\"
        B = obs.predicate_states.shape[0]

        # Default IDs if not provided (backward compat)
        type_ids = obs.predicate_type_ids if obs.predicate_type_ids is not None else jnp.zeros((B, MAX_NUM_PREDICATES), dtype=jnp.int32)
        name_ids = obs.predicate_name_ids if obs.predicate_name_ids is not None else jnp.zeros((B, MAX_NUM_PREDICATES), dtype=jnp.int32)
        arg_ids = obs.predicate_arg_ids if obs.predicate_arg_ids is not None else jnp.zeros((B, MAX_NUM_PREDICATES), dtype=jnp.int32)
        progress = obs.predicate_progress if obs.predicate_progress is not None else obs.predicate_states.astype(jnp.float32)

        # Layer 1: Build per-predicate features and encode with shared phi
        type_feat = self.pred_type_emb(type_ids)    # [B, P, 64]
        name_feat = self.pred_name_emb(name_ids)    # [B, P, 128]
        arg_feat = self.pred_arg_emb(arg_ids)        # [B, P, 128]
        progress_feat = progress[..., None]           # [B, P, 1]
        satisfied_feat = obs.predicate_states.astype(jnp.float32)[..., None]  # [B, P, 1]

        x = jnp.concatenate([type_feat, name_feat, arg_feat, progress_feat, satisfied_feat], axis=-1)  # [B, P, 322]
        h = nnx.relu(self.phi_fc1(x))    # [B, P, 512]
        phi_out = self.phi_fc2(h)         # [B, P, 1024]

        # Layer 2: Per-type modules with progress
        progress_cat = progress[..., None]  # [B, P, 1]
        emb_with_progress = jnp.concatenate([phi_out, progress_cat], axis=-1)  # [B, P, 1025]

        forall_out = nnx.relu(self.forall_fc1(emb_with_progress))
        forall_out = self.forall_fc2(forall_out) + phi_out  # residual

        exists_out = nnx.relu(self.exists_fc1(emb_with_progress))
        exists_out = self.exists_fc2(exists_out) + phi_out  # residual

        # Select by type: 0=atomic->phi_out, 1=forall->forall_out, 2=exists->exists_out
        is_forall = (type_ids == 1)[..., None]  # [B, P, 1]
        is_exists = (type_ids == 2)[..., None]
        pred_embs = jnp.where(is_forall, forall_out, jnp.where(is_exists, exists_out, phi_out))

        # Layer 3: Masked sum aggregation (Deep Sets)
        mask = obs.predicate_mask[..., None].astype(jnp.float32)  # [B, P, 1]
        summed = jnp.sum(pred_embs * mask, axis=1)  # [B, 1024]

        # rho MLP
        rho_h = nnx.relu(self.rho_fc1(summed))  # [B, 1024]
        predicate_repr = self.rho_fc2(rho_h)     # [B, 2048]

        # Project to 4 tokens to match V1 structure
        tokens_flat = self.pred_token_proj(predicate_repr)  # [B, 4*2048]
        pred_tokens = tokens_flat.reshape(B, 4, -1)         # [B, 4, 2048]

        return pred_tokens
"""


# --- CHANGE 3: Modify embed_prefix (lines 592-621) ---
EMBED_PREFIX_CHANGE = """
# Replace lines 599-612 in embed_prefix with:

            if obs.predicate_states is not None and obs.predicate_mask is not None:
                # V2 Deep Sets encoder
                pred_tokens = self.encode_predicates_deep_sets(obs)  # [B, 4, 2048]
            else:
                raise ValueError("predicate_states and predicate_mask must be provided")

            # Task token sequence: [base_task, pred_tok_0, pred_tok_1, pred_tok_2, pred_tok_3]
            task_sequence = jnp.concatenate([
                base_task_embedding[:, None, :],  # [B, 1, 2048]
                pred_tokens                        # [B, 4, 2048]
            ], axis=1)  # [B, 5, 2048]

            # Rest unchanged (task_mask, ar_mask same as V1)
"""


# --- CHANGE 4: Loss additions (same as Exp 2) ---
LOSS_ADDITION = """
        # V2: Progress regression loss
        if train and observation.predicate_progress is not None:
            progress_pred = jax.nn.sigmoid(self.progress_pred_from_vlm(base_task_output))
            progress_target = observation.predicate_progress.astype(jnp.float32)
            progress_mse = jnp.square(progress_pred - progress_target) * pred_mask * valid_pred_mask.astype(jnp.float32)
            num_valid_progress = jnp.maximum(jnp.sum(pred_mask * valid_pred_mask.astype(jnp.float32), axis=-1), 1.0)
            per_sample_progress_loss = jnp.sum(progress_mse, axis=-1) / num_valid_progress
            losses["progress_loss"] = jnp.mean(per_sample_progress_loss)
            progress_loss_value = self.config.progress_loss_weight * losses["progress_loss"]
        else:
            progress_loss_value = 0.0

        losses["total_loss"] = losses["action_loss"] + fast_loss_value + predicate_loss_value + progress_loss_value
"""
