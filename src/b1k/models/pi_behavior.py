"""The main model for BEHAVIOR-1K challenge.

Based on Pi0.5 implementation from PhysicalIntelligence/openpi
"""

import logging
import pathlib

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import gemma as _gemma
from openpi.models import siglip as _siglip
from openpi.models.pi0 import make_attn_mask, posemb_sincos
from openpi.shared import array_typing as at

# Import from our custom modules
from b1k.models import pi_behavior_config
from b1k.models.observation import Observation, preprocess_observation
from b1k.models.pi_behavior_config import (
    TASK_NUM_PREDICATES,
    MAX_NUM_PREDICATES,
    TOTAL_TASK_PREDICATE_EMBEDDINGS,
    TASK_PREDICATE_OFFSETS,
)

logger = logging.getLogger("b1k")


class KVCacheTransform(nnx.Module):
    """Transforms prefix KV cache by mixing across layers.
    
    Each destination layer's K and V become learnable linear combinations
    of all source layers' K and V, plus a bias term. This allows the action
    expert to attend to learned combinations of VLM layers rather than being
    forced to attend layer-by-layer.
    
    Initialized as identity transform (k_coeffs = I, bias = 0) so the model
    starts with the same behavior as without transformation.
    """
    
    def __init__(self, num_layers: int, head_dim: int, num_kv_heads: int, rngs: nnx.Rngs):
        # K transformation: [dest_layer, src_layer]
        # Initialize as identity so transformation is initially a no-op
        self.k_coeffs = nnx.Param(jnp.eye(num_layers, dtype=jnp.float32))
        
        # K bias: [layer, num_kv_heads, head_dim]
        # Initialize as zeros
        self.k_bias = nnx.Param(jnp.zeros((num_layers, num_kv_heads, head_dim), dtype=jnp.float32))
        
        # V transformation (independent from K)
        self.v_coeffs = nnx.Param(jnp.eye(num_layers, dtype=jnp.float32))
        self.v_bias = nnx.Param(jnp.zeros((num_layers, num_kv_heads, head_dim), dtype=jnp.float32))
    
    def __call__(self, kv_cache: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Transform KV cache by mixing across layers.
        
        Args:
            kv_cache: Tuple of (cache_k, cache_v) where each has shape
                     [num_layers, batch, seq_len, num_kv_heads, head_dim]
        
        Returns:
            Transformed (k_new, v_new) with same shape and dtype as input
        """
        cache_k, cache_v = kv_cache
        # Shape: [layers, batch, seq_len, num_kv_heads, head_dim]
        
        # Preserve original dtype (important for bfloat16 training)
        original_dtype = cache_k.dtype
        
        # Transform K: each destination layer is a weighted combination of all source layers
        # k_new[dest] = sum_src(k_coeffs[dest, src] * cache_k[src]) + k_bias[dest]
        # Einsum: [dest, src] @ [src, batch, seq, heads, dim] -> [dest, batch, seq, heads, dim]
        k_new = jnp.einsum('ds,sbtkh->dbtkh', self.k_coeffs.value, cache_k)
        k_new = k_new + self.k_bias.value[:, None, None, :, :]  # Add bias
        
        # Transform V (same operation, independent parameters)
        v_new = jnp.einsum('ds,sbtkh->dbtkh', self.v_coeffs.value, cache_v)
        v_new = v_new + self.v_bias.value[:, None, None, :, :]
        
        # Cast back to original dtype
        k_new = k_new.astype(original_dtype)
        v_new = v_new.astype(original_dtype)
        
        return (k_new, v_new)


class PiBehavior(_model.BaseModel):
    def __init__(self, config: pi_behavior_config.PiBehaviorConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        
        # Store config for later use
        self.config = config
        
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        
        # Initialize Gemma models with AdaRMS (Pi05 style)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        
        # Initialize vision model
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        
        # KV cache transformation for cross-layer attention
        # Allows action expert to attend to learned combinations of VLM layers
        if config.use_kv_transform:
            self.kv_transform = KVCacheTransform(
                num_layers=paligemma_config.depth,
                head_dim=paligemma_config.head_dim,
                num_kv_heads=paligemma_config.num_kv_heads,
                rngs=rngs
            )
        else:
            self.kv_transform = None
        
        # Task embeddings table - trainable embeddings for each task
        self.task_embeddings = nnx.Embed(
            num_embeddings=config.num_tasks,
            features=config.task_embedding_dim,
            rngs=rngs,
        )
        
        # Predicate predictor - predicts multi-label binary predicates from VLM output
        # Outputs MAX_NUM_PREDICATES logits, sigmoid applied per predicate (independent binary)
        self.predicate_pred_from_vlm = nnx.Linear(paligemma_config.width, MAX_NUM_PREDICATES, rngs=rngs)
        
        # Task + predicate fusion layers
        self.predicate_encoding_dim = config.task_embedding_dim // 2  # Half of task embedding dim (1024)
        
        # Task-specific predicate embeddings (one per predicate per task)
        # Total embeddings = sum of predicates across all tasks (233 total)
        # Each predicate embedding represents the meaning of that object/item being "done"
        self.task_predicate_embeddings = nnx.Embed(
            num_embeddings=TOTAL_TASK_PREDICATE_EMBEDDINGS,
            features=self.predicate_encoding_dim,
            rngs=rngs,
        )
        
        # Gated fusion layers
        # Input: task_embedding + sincos + task_predicate_emb = task_dim + 2*pred_dim
        fusion_input_dim = config.task_embedding_dim + 2 * self.predicate_encoding_dim
        
        # Gate networks to learn how to combine different signals
        self.gate_sincos = nnx.Linear(fusion_input_dim, self.predicate_encoding_dim, rngs=rngs)
        self.gate_predicate = nnx.Linear(fusion_input_dim, self.predicate_encoding_dim, rngs=rngs)
        self.gate_task = nnx.Linear(fusion_input_dim, config.task_embedding_dim, rngs=rngs)
        
        # Fusion networks to create multiple conditioned vectors
        self.fusion_layer1 = nnx.Linear(fusion_input_dim, config.task_embedding_dim * 2, rngs=rngs)
        self.fusion_layer2 = nnx.Linear(config.task_embedding_dim * 2, config.task_embedding_dim, rngs=rngs)
        
        # Additional projection for remaining-focus representation
        self.predicate_projection = nnx.Linear(2 * self.predicate_encoding_dim, config.task_embedding_dim, rngs=rngs)
        
        # Pi05 style layers
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Correlated noise generation
        # Initialize as NNX Intermediate (excluded from checkpoints, loaded from norm_stats)
        # Full correlation matrix with beta shrinkage for robustness
        flat_dim = config.action_horizon * config.action_dim
        self.action_correlation_cholesky = nnx.Intermediate(
            jnp.eye(flat_dim),  # Identity matrix as placeholder
        )
        self.correlation_loaded = False  # Track if correlation matrix has been loaded
        self.use_correlated_noise = config.use_correlated_noise
        self.correlation_beta = config.correlation_beta  # Shrinkage parameter for regularization
        
        # Inpainting cache: stores precomputed matrices for simple correlation-based inpainting
        # Key: num_inpainted_steps (length of inpainted sequence)
        # Value: dict with {O_indices, U_indices, Sigma_UO_SOOinv}
        self.inpainting_cache = {}
        
        # FAST auxiliary training components
        if config.use_fast_auxiliary:
            # FAST embedding layer (vocab_size → paligemma_width)
            # Use paligemma width (2048) to match other prefix tokens
            self.fast_token_embedding = nnx.Embed(
                num_embeddings=config.fast_vocab_size,
                features=paligemma_config.width,
                rngs=rngs
            )
            
            # FAST projection head (paligemma_width → vocab_size)
            self.fast_token_proj = nnx.Linear(
                paligemma_config.width,
                config.fast_vocab_size,
                rngs=rngs
            )
            
            logger.info(f"FAST auxiliary enabled, vocab_size={config.fast_vocab_size}")

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def load_correlation_matrix(self, norm_stats: dict):
        """Load full correlation matrix from normalization statistics and apply shrinkage.
        
        This should be called after model initialization when norm_stats are available.
        Applies shrinkage regularization: S_reg = beta * S + (1-beta) * I for robustness.
        
        Args:
            norm_stats: Dictionary containing normalization statistics (from normalize.load()),
                       with 'actions' key containing NormStats with action_correlation_cholesky field.
                       
        Raises:
            ValueError: If use_correlated_noise=True but correlation matrix is missing.
            TypeError: If norm_stats structure is incorrect.
        """
        if not self.use_correlated_noise:
            logger.info("Correlated noise disabled in config, skipping correlation matrix loading")
            return
        
        # Validate norm_stats is a dict
        if not isinstance(norm_stats, dict):
            raise TypeError(
                f"norm_stats must be a dict, got {type(norm_stats).__name__}. "
                "Ensure norm_stats are loaded using openpi.shared.normalize.load()."
            )
        
        # Check 'actions' key exists
        if 'actions' not in norm_stats:
            raise ValueError(
                "use_correlated_noise=True but 'actions' key not found in norm_stats. "
                f"Found keys: {list(norm_stats.keys())}. "
                "Run compute_norm_stats.py with --correlation flag to generate correlation matrix."
            )
        
        actions_stats = norm_stats['actions']
        
        # Extract correlation matrix (support both dict and attribute access for flexibility)
        if isinstance(actions_stats, dict):
            chol_matrix = actions_stats.get('action_correlation_cholesky')
            access_method = "dict"
        elif hasattr(actions_stats, 'action_correlation_cholesky'):
            chol_matrix = actions_stats.action_correlation_cholesky
            access_method = "attribute"
        else:
            raise TypeError(
                f"norm_stats['actions'] has unexpected type {type(actions_stats).__name__} "
                f"and cannot access 'action_correlation_cholesky'. "
                "Ensure norm_stats are loaded using openpi.shared.normalize.load()."
            )
        
        # Strict validation: correlation matrix must exist and be non-None
        if chol_matrix is None:
            raise ValueError(
                "use_correlated_noise=True but 'action_correlation_cholesky' is None in norm_stats['actions']. "
                "This means the correlation matrix was not computed during norm_stats generation. "
                "Run compute_norm_stats.py with --correlation flag to generate correlation matrix."
            )
        
        logger.info(f"Successfully accessed correlation matrix via {access_method} access")
        
        # Validate correlation matrix shape
        expected_dim = self.action_horizon * self.action_dim
        try:
            L = jnp.array(chol_matrix)
        except Exception as e:
            raise ValueError(
                f"Failed to convert action_correlation_cholesky to array: {e}. "
                "The correlation matrix may be corrupted or in an invalid format."
            )
        
        if L.ndim != 2 or L.shape[0] != L.shape[1]:
            raise ValueError(
                f"action_correlation_cholesky must be a square 2D matrix, got shape {L.shape}. "
                f"Expected shape: ({expected_dim}, {expected_dim})"
            )
        
        if L.shape[0] != expected_dim:
            raise ValueError(
                f"action_correlation_cholesky has wrong dimensions: {L.shape[0]}x{L.shape[0]}. "
                f"Expected {expected_dim}x{expected_dim} (action_horizon={self.action_horizon} * action_dim={self.action_dim}). "
                "This indicates the correlation matrix was computed for a different action space configuration."
            )
        
        # Reconstruct covariance matrix from Cholesky
        Sigma = L @ L.T
        
        # Apply shrinkage regularization: Σ_reg = beta * Σ + (1-beta) * I
        beta = self.correlation_beta
        logger.info(f"Applying shrinkage regularization with beta={beta:.2f}")
        
        Sigma_reg = beta * Sigma + (1 - beta) * jnp.eye(Sigma.shape[0])
        
        # Compute Cholesky decomposition of regularized covariance
        try:
            L_reg = jnp.linalg.cholesky(Sigma_reg)
        except Exception as e:
            raise RuntimeError(
                f"Cholesky decomposition failed on regularized covariance: {e}. "
                "This indicates the regularized correlation matrix is not positive definite. "
                f"Current beta={beta:.2f}. Try decreasing correlation_beta closer to 0.0 for more shrinkage/regularization."
            )
        
        # Update the Intermediate value
        self.action_correlation_cholesky.value = L_reg
        self.correlation_loaded = True
        
        logger.info(
            f"✓ Loaded correlation matrix with shape {L_reg.shape} "
            f"(beta={beta:.2f} shrinkage applied)"
        )
        logger.info(
            f"  Memory usage: {L_reg.nbytes / 1024 / 1024:.2f} MB"
        )
    
    def generate_correlated_noise(
        self, 
        rng: at.KeyArrayLike, 
        batch_size: int,
    ) -> at.Float[at.Array, "b {self.action_horizon} {self.action_dim}"]:
        """Generate correlated noise matching action covariance structure.
        
        Uses full correlation matrix with optional beta shrinkage for robustness.
        
        Args:
            rng: Random key for noise generation
            batch_size: Number of noise samples to generate
            
        Returns:
            Correlated noise with shape [batch_size, action_horizon, action_dim]
            
        Raises:
            RuntimeError: If use_correlated_noise=True but correlation matrix not loaded.
        """
        if not self.use_correlated_noise:
            # Independent Gaussian noise when correlated noise is disabled
            return jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        
        if not self.correlation_loaded:
            raise RuntimeError(
                "use_correlated_noise=True but correlation matrix is not loaded. "
                "Ensure load_correlation_matrix() was called during model initialization. "
                "Run compute_norm_stats.py with --correlation flag to generate correlation matrix."
            )
        
        # Generate standard correlated noise using Cholesky decomposition
        flat_dim = self.action_horizon * self.action_dim
        standard_normal = jax.random.normal(rng, (batch_size, flat_dim))
        correlated_flat = standard_normal @ self.action_correlation_cholesky.value.T
        correlated_noise = correlated_flat.reshape(batch_size, self.action_horizon, self.action_dim)
        return correlated_noise

    def _precompute_correction_matrix(
        self,
        O_indices: at.Int[at.Array, " nO"],
        U_indices: at.Int[at.Array, " nU"],
    ) -> dict:
        """Precompute matrix for correlation-aware inpainting correction.
        
        Computes Σ_{UO}Σ_{OO}^{-1} which propagates corrections from O to U
        while preserving correlation structure.
        
        Args:
            O_indices: Flat indices of inpainted dimensions [|O|]
            U_indices: Flat indices of free dimensions [|U|]
            
        Returns:
            Dictionary with {O_indices, U_indices, correction_matrix}
            
        Raises:
            RuntimeError: If correlation matrix is not loaded
        """
        if not self.correlation_loaded:
            raise RuntimeError(
                "Cannot precompute correction matrix: correlation matrix not loaded. "
                "Call load_correlation_matrix() first."
            )
        
        L = self.action_correlation_cholesky.value
        Sigma = L @ L.T  # Full covariance matrix [hd, hd]
        
        # Extract submatrices
        Sigma_OO = Sigma[jnp.ix_(O_indices, O_indices)]  # [|O|, |O|]
        Sigma_UO = Sigma[jnp.ix_(U_indices, O_indices)]  # [|U|, |O|]
        
        # Compute correction matrix: Σ_{UO} @ Σ_{OO}^{-1}
        # This propagates corrections from O to U
        eps_OO = 1e-6 * jnp.maximum(jnp.mean(jnp.diag(Sigma_OO)), 1.0)
        Sigma_OO_reg = Sigma_OO + eps_OO * jnp.eye(Sigma_OO.shape[0])
        
        # Solve Σ_{OO}_reg @ X = Σ_{UO}.T for X, then transpose
        correction_matrix = jax.scipy.linalg.solve(
            Sigma_OO_reg, Sigma_UO.T, assume_a='pos'
        ).T  # [|U|, |O|]
        
        return {
            'O_indices': O_indices,
            'U_indices': U_indices,
            'correction_matrix': correction_matrix,  # Σ_{UO}Σ_{OO}^{-1}
        }

    def encode_predicate_progress(
        self, 
        predicate_states: at.Bool[at.Array, "b p"],
        predicate_mask: at.Bool[at.Array, "b p"],
    ) -> at.Float[at.Array, "b {self.predicate_encoding_dim}"]:
        """Encode predicate progress as sincos positional encoding.
        
        Progress = num_done_predicates / num_total_predicates
        
        Args:
            predicate_states: [B, P] True = predicate is done
            predicate_mask: [B, P] True = predicate is valid for this task
            
        Returns:
            Positional encodings [B, 1024]
        """
        # Count done and total predicates
        num_done = jnp.sum(predicate_states & predicate_mask, axis=-1).astype(jnp.float32)  # [B]
        num_total = jnp.sum(predicate_mask, axis=-1).astype(jnp.float32)  # [B]
        
        # Progress: 0.0 = nothing done, 1.0 = all done
        progress = num_done / jnp.maximum(num_total, 1.0)  # [B]
        
        return posemb_sincos(
            progress, 
            self.predicate_encoding_dim, 
            min_period=1e-3, 
            max_period=1.0
        )

    def aggregate_predicate_embeddings(
        self,
        task_ids: at.Int[at.Array, " b"],
        predicate_states: at.Bool[at.Array, "b p"],
        predicate_mask: at.Bool[at.Array, "b p"],
    ) -> tuple[at.Float[at.Array, "b d"], at.Float[at.Array, "b d"]]:
        """Aggregate predicate embeddings into done and remaining representations.
        
        Args:
            task_ids: [B] Task IDs for task-specific predicate embeddings
            predicate_states: [B, P] True = predicate is done
            predicate_mask: [B, P] True = predicate is valid for this task
            
        Returns:
            done_agg: [B, 1024] Mean of done predicate embeddings
            remaining_agg: [B, 1024] Mean of remaining predicate embeddings
        """
        batch_size = task_ids.shape[0]
        
        # Get all predicate embeddings for each sample
        # We need to gather embeddings for all predicates of each task
        task_pred_offsets = jnp.array(TASK_PREDICATE_OFFSETS, dtype=jnp.int32)
        task_num_preds = jnp.array(TASK_NUM_PREDICATES, dtype=jnp.int32)
        
        offsets = task_pred_offsets[task_ids]  # [B]
        num_preds = task_num_preds[task_ids]   # [B]
        
        # Build indices for all predicates: [B, MAX_NUM_PREDICATES]
        pred_range = jnp.arange(MAX_NUM_PREDICATES)  # [P]
        pred_indices = offsets[:, None] + pred_range[None, :]  # [B, P]
        
        # Clamp indices to valid range (for predicates beyond task's count)
        max_idx = TOTAL_TASK_PREDICATE_EMBEDDINGS - 1
        pred_indices = jnp.clip(pred_indices, 0, max_idx)
        
        # Get all embeddings [B, P, 1024]
        all_embeddings = self.task_predicate_embeddings(pred_indices)
        
        # Create done and remaining masks
        done_mask = predicate_states & predicate_mask  # [B, P]
        remaining_mask = ~predicate_states & predicate_mask  # [B, P]
        
        # Mean pooling with masks
        # Done embeddings
        done_sum = jnp.sum(all_embeddings * done_mask[..., None], axis=1)  # [B, 1024]
        num_done = jnp.maximum(jnp.sum(done_mask, axis=1, keepdims=True), 1.0)  # [B, 1]
        done_agg = done_sum / num_done  # [B, 1024]
        
        # Remaining embeddings
        remaining_sum = jnp.sum(all_embeddings * remaining_mask[..., None], axis=1)  # [B, 1024]
        num_remaining = jnp.maximum(jnp.sum(remaining_mask, axis=1, keepdims=True), 1.0)  # [B, 1]
        remaining_agg = remaining_sum / num_remaining  # [B, 1024]
        
        return done_agg, remaining_agg

    def fuse_task_and_predicates(
        self, 
        task_embedding: at.Float[at.Array, "b d"], 
        task_ids: at.Int[at.Array, " b"], 
        predicate_states: at.Bool[at.Array, "b p"],
        predicate_mask: at.Bool[at.Array, "b p"],
    ) -> at.Float[at.Array, "b n d"]:
        """Fuse task embedding with predicate states using multiple representations.
        
        Uses multi-label predicates where each predicate indicates if an object is done.
        
        Returns multiple vectors differently conditioned by predicate states:
        1. Task-gated representation (task embedding modulated by predicates)
        2. Balanced fusion (task + predicates combined)
        3. Remaining-focus representation (what to manipulate next)
        4. Done-focus representation (what to avoid)
        
        All output representations have dimension 2048 (task_embedding_dim).
        
        Args:
            task_embedding: Base task embedding [b, 2048]
            task_ids: Task IDs for task-specific predicate embeddings [b]
            predicate_states: [B, P] True = object is done
            predicate_mask: [B, P] True = predicate is valid for this task
            
        Returns:
            Multiple fused embeddings [b, 4, 2048]
        """
        # Get predicate representations
        progress_encoding = self.encode_predicate_progress(predicate_states, predicate_mask)  # [b, 1024]
        done_agg, remaining_agg = self.aggregate_predicate_embeddings(
            task_ids, predicate_states, predicate_mask
        )  # [b, 1024], [b, 1024]
        
        # Concatenate inputs for gating: task (2048) + progress (1024) + done (1024) + remaining (1024) = 5120
        # But we need to match the existing fusion_input_dim = 4096
        # So we use: task (2048) + progress (1024) + (done + remaining mean) (1024) = 4096
        combined_predicate_agg = (done_agg + remaining_agg) / 2.0  # [b, 1024]
        
        all_inputs = jnp.concatenate([
            task_embedding,        # [b, 2048]
            progress_encoding,     # [b, 1024]
            combined_predicate_agg # [b, 1024]
        ], axis=-1)  # [b, 4096]
        
        # Learn gates for each component (sigmoid to get 0-1 scaling)
        gate_progress = nnx.sigmoid(self.gate_sincos(all_inputs))      # [b, 1024]
        gate_predicate = nnx.sigmoid(self.gate_predicate(all_inputs))  # [b, 1024]
        gate_task = nnx.sigmoid(self.gate_task(all_inputs))            # [b, 2048]
        
        # 1. Task-gated representation: task embedding modulated by predicate info [b, 2048]
        task_gated = task_embedding * gate_task
        
        # 2. Balanced fusion: combine all signals through fusion network [b, 2048]
        x = self.fusion_layer1(all_inputs)  # [b, 4096]
        x = nnx.relu(x)
        balanced_fusion = self.fusion_layer2(x)  # [b, 2048]
        
        # 3. Remaining-focus: what to manipulate next [b, 2048]
        gated_remaining = jnp.concatenate([
            progress_encoding * gate_progress,  # [b, 1024]
            remaining_agg * gate_predicate      # [b, 1024]
        ], axis=-1)  # [b, 2048]
        remaining_focus = self.predicate_projection(gated_remaining)  # [b, 2048]
        
        # 4. Done-focus: what to avoid (pure done + progress) [b, 2048]
        done_focus = jnp.concatenate([progress_encoding, done_agg], axis=-1)  # [b, 2048]
        
        # Stack all four representations [b, 4, 2048]
        fused_embeddings = jnp.stack([task_gated, balanced_fusion, remaining_focus, done_focus], axis=1)
        
        return fused_embeddings

    @at.typecheck
    def embed_prefix(
        self, 
        obs: Observation
    ) -> tuple[
        at.Float[at.Array, "b s emb"], 
        at.Bool[at.Array, "b s"], 
        at.Bool[at.Array, " s"]
    ]:
        """
        Embed prefix: images + task + state + FAST_tokens (if provided).
        
        Args:
            obs: Observation (may include fast_tokens and fast_token_mask)
            
        Returns:
            tokens, input_mask, ar_mask
        """
        input_mask = []
        ar_mask = []
        tokens = []
        
        # Embed images
        image_token_list = []
        # Respect freeze_vision_backbone config: if frozen, always use train=False
        # If not frozen, use the model's training state (self.deterministic)
        vision_train_mode = (not self.deterministic) and (not self.config.freeze_vision_backbone)
        
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=vision_train_mode)
            image_token_list.append(image_tokens)  # Store for predicate prediction

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # Image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # Add task embeddings with predicate state fusion
        if obs.tokenized_prompt is not None:
            # obs.tokenized_prompt now contains task_ids (shape: [batch_size, 2])
            task_ids = obs.tokenized_prompt[:, 0]  # Extract task_id: [batch_size]
            base_task_embedding = self.task_embeddings(task_ids)  # shape: [batch_size, embed_dim]
            
            # Check if predicate states are available (required for PI_BEHAVIOR)
            if obs.predicate_states is not None and obs.predicate_mask is not None:
                # Use predicate-based fusion
                fused_task_embeddings = self.fuse_task_and_predicates(
                    base_task_embedding, task_ids, 
                    obs.predicate_states, obs.predicate_mask
                )
            else:
                raise ValueError("predicate_states and predicate_mask must be provided for PI_BEHAVIOR model")
            
            # Create task token sequence: [base_task, fused_0, fused_1, fused_2, fused_3]
            task_sequence = jnp.concatenate([
                base_task_embedding[:, None, :],  # [b, 1, d] - base task token
                fused_task_embeddings              # [b, 4, d] - predicate-conditioned tokens
            ], axis=1)  # [b, 5, d]
            
            tokens.append(task_sequence)
            # All task tokens are valid
            task_mask = jnp.ones((obs.tokenized_prompt.shape[0], 5), dtype=jnp.bool_)
            input_mask.append(task_mask)
            # Hierarchical attention: base task (False) then predicate tokens (True, False, False, False)
            # Base task attends to images bidirectionally
            # Predicate tokens attend to images+task but not vice versa
            ar_mask += [False] + [True, False, False, False]
            
        # Add state as discrete tokens (Pi05 style)
        # Discretize state into bins
        discretized_state = jnp.digitize(obs.state, bins=jnp.linspace(-1, 1, 256 + 1)[:-1]) - 1
        discretized_state = jnp.clip(discretized_state, 0, 255)  # Ensure valid range
        
        # Embed each dimension of the discretized state
        state_tokens = []
        for i in range(obs.state.shape[-1]):
            state_dim_tokens = self.PaliGemma.llm(discretized_state[:, i:i+1], method="embed")
            state_tokens.append(state_dim_tokens)
        
        if state_tokens:
            state_tokens = jnp.concatenate(state_tokens, axis=1)  # shape: [batch_size, state_dim, embed_dim]
            tokens.append(state_tokens)
            input_mask.append(jnp.ones((obs.state.shape[0], obs.state.shape[-1]), dtype=jnp.bool_))
            # State tokens have full bidirectional attention with all prefix tokens
            # (images, task, predicates, and other state tokens)
            ar_mask += [False] * state_tokens.shape[1]
        
        # FAST tokens (from observation if provided)
        if self.config.use_fast_auxiliary and obs.fast_tokens is not None:
            fast_tokens = obs.fast_tokens  # [B, T]
            fast_token_mask = obs.fast_token_mask  # [B, T]
            
            # Teacher forcing: shift right [BOS, tok0, tok1, ..., tok_{T-1}]
            bos_token = jnp.zeros((fast_tokens.shape[0], 1), dtype=jnp.int32)
            shifted_tokens = jnp.concatenate([bos_token, fast_tokens[:, :-1]], axis=1)
            
            # Shift mask too: [True, mask_0, mask_1, ..., mask_{T-1}]
            bos_mask = jnp.ones((fast_tokens.shape[0], 1), dtype=jnp.bool_)
            shifted_mask = jnp.concatenate([bos_mask, fast_token_mask[:, :-1]], axis=1)
            
            # Embed using FAST embedding layer (NOT Paligemma!)
            fast_token_emb = self.fast_token_embedding(shifted_tokens)  # [B, T, D]
            
            tokens.append(fast_token_emb)
            input_mask.append(shifted_mask)  # Use the actual token mask
            # Causal for FAST: ALL tokens are causal (pure autoregressive)
            ar_mask += [True] * shifted_tokens.shape[1]
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"],
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        
        # Pi05 style: no explicit state token in suffix (it's in prefix as discrete tokens)
        
        action_tokens = self.action_in_proj(noisy_actions)
        # Embed timestep using sine-cosine positional encoding
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        
        # Pi05 style: time MLP for adaRMS
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        action_expert_tokens = action_tokens
        adarms_cond = time_emb
        
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        
        # image/task/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """Not used - we only use compute_detailed_loss() for training."""
        raise NotImplementedError("Use compute_detailed_loss() instead")

    @override
    def compute_detailed_loss(
        self, rng: at.KeyArrayLike, observation: Observation, actions: _model.Actions, *, train: bool = False, num_flow_samples: int = 1
    ) -> dict[str, at.Float[at.Array, "*b"]]:
        """
        Compute detailed loss with multiple flow matching samples.
        
        Simplified approach using KV cache:
        - Compute prefix KV cache once (with FAST tokens)
        - Remove FAST tokens from cache (action expert doesn't attend to FAST)
        - Process N flow samples independently, each reusing the same cached prefix
        - Each sample has different noise and different time
        - Average losses across samples
        """
        losses = {}

        preprocess_rng, rng = jax.random.split(rng)
        observation = preprocess_observation(preprocess_rng, observation, train=train)

        batch_size = actions.shape[0]
        
        # 1. Embed prefix once (includes FAST tokens if provided in observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        
        # 2. Compute prefix KV cache
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache_full = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions_prefix
        )
        
        # 3. Predict predicates from VLM output of base task token
        # Base task token is the first token after all image tokens
        # Image tokens all have ar_mask=False, task starts with ar_mask=False (base) then True (predicate tokens)
        # Structure: [images (all False)] [base_task (False)] [predicate_tokens (True, False, False, False)]
        # Find first True (first predicate token), base task is at that index - 1
        first_predicate_token_idx = jnp.argmax(prefix_ar_mask)  # Returns index of first True
        base_task_token_idx = first_predicate_token_idx - 1
        base_task_output = prefix_out[:, base_task_token_idx, :]
        
        # Multi-label predicate prediction
        predicate_logits = self.predicate_pred_from_vlm(base_task_output)  # [B, MAX_NUM_PREDICATES]
        task_ids = observation.tokenized_prompt[:, 0]  # [B]
        
        # Mask out invalid predicates for each task
        task_num_preds_array = jnp.array(TASK_NUM_PREDICATES, dtype=jnp.int32)
        task_num_preds = task_num_preds_array[task_ids]  # [B]
        pred_range = jnp.arange(MAX_NUM_PREDICATES)  # [20]
        valid_pred_mask = pred_range[None, :] < task_num_preds[:, None]  # [B, 20]
        
        # 4. Extract FAST loss from prefix output (before removing from cache)
        fast_loss_value = 0.0
        fast_len = 0
        fast_targets = observation.fast_tokens
        fast_token_mask = observation.fast_token_mask
        
        if self.config.use_fast_auxiliary and fast_targets is not None:
            fast_len = fast_targets.shape[1]
            fast_start_idx = prefix_tokens.shape[1] - fast_len
            fast_outputs = prefix_out[:, fast_start_idx:, :]  # [B, T, D]
            
            # Project to FAST vocab
            fast_logits = self.fast_token_proj(fast_outputs)  # [B, T, vocab_size]
            
            # Cross-entropy loss with teacher forcing
            pred_logits = fast_logits  # [B, T, vocab]
            target_tokens = fast_targets  # [B, T]
            loss_mask = fast_token_mask  # [B, T]
            
            log_probs = jax.nn.log_softmax(pred_logits, axis=-1)
            target_log_probs = jnp.take_along_axis(
                log_probs,
                target_tokens[:, :, None],
                axis=-1
            ).squeeze(-1)  # [B, T]
            
            fast_token_loss = -target_log_probs  # [B, T]
            
            # Apply mask and normalize by number of valid tokens
            masked_loss = fast_token_loss * loss_mask  # [B, T]
            num_valid_tokens = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1)  # [B]
            losses["fast_loss"] = jnp.sum(masked_loss, axis=-1) / num_valid_tokens  # [B]
            
            # Accuracy (only on valid tokens)
            pred_tokens = jnp.argmax(pred_logits, axis=-1)
            correct = (pred_tokens == target_tokens) * loss_mask
            losses["fast_accuracy"] = jnp.sum(correct, axis=-1) / num_valid_tokens
            
            fast_loss_value = self.config.fast_loss_weight * jnp.mean(losses["fast_loss"])
        elif fast_targets is not None:
            # FAST auxiliary is disabled but data contains FAST tokens
            raise ValueError(
                "use_fast_auxiliary=False but observation contains fast_tokens. "
                "Either enable use_fast_auxiliary in config or ensure data doesn't contain fast_tokens."
            )
        
        # 5. Remove FAST tokens from KV cache (action expert doesn't attend to FAST)
        # KV cache shape: [layers, batch, seq_len, num_kv_heads, head_dim]
        if fast_len > 0:
            cache_k, cache_v = kv_cache_full
            # Remove last fast_len tokens from sequence dimension
            cache_k = cache_k[:, :, :-fast_len, :, :]
            cache_v = cache_v[:, :, :-fast_len, :, :]
            kv_cache_for_actions = (cache_k, cache_v)
            prefix_len_for_actions = prefix_tokens.shape[1] - fast_len
            # Truncate prefix mask and ar_mask for action expert
            prefix_mask_for_actions = prefix_mask[:, :-fast_len]
            prefix_ar_mask_for_actions = prefix_ar_mask[:-fast_len]
        else:
            kv_cache_for_actions = kv_cache_full
            prefix_len_for_actions = prefix_tokens.shape[1]
            prefix_mask_for_actions = prefix_mask
            prefix_ar_mask_for_actions = prefix_ar_mask
        
        # 6. Knowledge insulation: stop gradients from action expert to VLM
        # This must happen BEFORE kv_transform so transform still receives gradients
        if self.config.use_knowledge_insulation:
            kv_cache_for_actions = jax.tree.map(jax.lax.stop_gradient, kv_cache_for_actions)
        
        # 7. Transform KV cache (after stop_gradient, so it receives action expert gradients)
        if self.kv_transform is not None:
            kv_cache_for_actions = self.kv_transform(kv_cache_for_actions)
        
        # 8. Define single flow sample processing
        def process_one_flow_sample(sample_rng):
            """Process one flow sample using the original cached prefix."""
            noise_rng, time_rng = jax.random.split(sample_rng)
            
            # Generate different noise and time for this sample
            noise = self.generate_correlated_noise(noise_rng, batch_size)
            time = jax.random.beta(time_rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001
            
            # Compute noisy actions and target velocity
            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions
            
            # Embed suffix for this sample
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time
            )
            
            # Build attention mask: suffix attends to prefix (without FAST) + itself
            # When using KV cache, mask shape should be [batch, suffix_len, prefix_len + suffix_len]
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(
                prefix_mask_for_actions, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            
            # Positions for suffix start after cached prefix
            suffix_positions = prefix_len_for_actions + jnp.cumsum(suffix_mask, axis=-1) - 1
            
            # Forward pass with cached prefix (discard returned cache - don't modify original!)
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache_for_actions,  # Original cache, reused for all samples
                adarms_cond=[None, adarms_cond]
            )
            
            # Compute velocity and loss
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            action_loss = jnp.square(v_t - u_t)  # [B, H, D]
            
            return action_loss
        
        # 9. Vectorize over N flow samples
        # Disable type checking inside vmap (jaxtyping doesn't handle traced values well)
        flow_rngs = jax.random.split(rng, num_flow_samples)
        with at.disable_typechecking():
            all_action_losses = jax.vmap(process_one_flow_sample)(flow_rngs)  # [N, B, H, D]
        
        # 10. Average over flow samples
        action_loss = jnp.mean(all_action_losses, axis=0)  # [B, H, D]
        
        # 11. Build per-dimension action losses
        # Base velocity (x,y,z)
        losses["action_loss_base_vel_x"] = jnp.mean(action_loss[..., 0], axis=-1)
        losses["action_loss_base_vel_y"] = jnp.mean(action_loss[..., 1], axis=-1)
        losses["action_loss_base_vel_z"] = jnp.mean(action_loss[..., 2], axis=-1)
        
        # Trunk joints (4)
        for i in range(4):
            losses[f"action_loss_trunk_{i}"] = jnp.mean(action_loss[..., 3+i], axis=-1)
            
        # Left arm joints (7)
        for i in range(7):
            losses[f"action_loss_left_arm_{i}"] = jnp.mean(action_loss[..., 7+i], axis=-1)
            
        # Left gripper
        losses["action_loss_left_gripper"] = jnp.mean(action_loss[..., 14], axis=-1)
        
        # Right arm joints (7)
        for i in range(7):
            losses[f"action_loss_right_arm_{i}"] = jnp.mean(action_loss[..., 15+i], axis=-1)
            
        # Right gripper
        losses["action_loss_right_gripper"] = jnp.mean(action_loss[..., 22], axis=-1)

        # Total action loss: mean over horizon (H) and action dims (D) -> [B]
        losses["action_loss"] = jnp.mean(action_loss, axis=(-2, -1))

        # 12. Add predicate loss during training (multi-label BCE)
        predicate_loss_value = 0.0
        if train and observation.predicate_states is not None and observation.predicate_mask is not None:
            # BCE loss: -[y*log(σ(x)) + (1-y)*log(1-σ(x))]
            # Using log_sigmoid for numerical stability
            gt_predicates = observation.predicate_states.astype(jnp.float32)  # [B, P]
            pred_mask = observation.predicate_mask.astype(jnp.float32)  # [B, P]
            
            # Binary cross entropy per predicate
            # log(σ(x)) = -softplus(-x), log(1-σ(x)) = -softplus(x)
            pos_loss = -jax.nn.log_sigmoid(predicate_logits)  # Loss when y=1
            neg_loss = -jax.nn.log_sigmoid(-predicate_logits)  # Loss when y=0 (i.e., log(1-sigmoid))
            bce_loss = gt_predicates * pos_loss + (1.0 - gt_predicates) * neg_loss  # [B, P]
            
            # Apply mask and compute mean over valid predicates
            masked_bce = bce_loss * pred_mask * valid_pred_mask.astype(jnp.float32)  # [B, P]
            num_valid = jnp.maximum(jnp.sum(pred_mask * valid_pred_mask.astype(jnp.float32), axis=-1), 1.0)  # [B]
            per_sample_loss = jnp.sum(masked_bce, axis=-1) / num_valid  # [B]
            
            losses["predicate_loss"] = jnp.mean(per_sample_loss)
            
            # Compute accuracy (threshold at 0.5, i.e., logits > 0)
            pred_binary = predicate_logits > 0  # [B, P]
            correct = (pred_binary == observation.predicate_states) * observation.predicate_mask * valid_pred_mask
            num_correct = jnp.sum(correct.astype(jnp.float32), axis=-1)
            losses["predicate_accuracy"] = jnp.mean(num_correct / num_valid)
            
            # Also track per-predicate-type accuracy
            pred_done = (pred_binary & observation.predicate_states) * observation.predicate_mask * valid_pred_mask
            gt_done = observation.predicate_states * observation.predicate_mask * valid_pred_mask
            num_gt_done = jnp.maximum(jnp.sum(gt_done.astype(jnp.float32), axis=-1), 1.0)
            losses["predicate_recall_done"] = jnp.mean(jnp.sum(pred_done.astype(jnp.float32), axis=-1) / num_gt_done)
            
            predicate_loss_value = self.config.predicate_loss_weight * jnp.mean(per_sample_loss)
        
        # 13. Total loss
        losses["total_loss"] = losses["action_loss"] + fast_loss_value + predicate_loss_value
        
        return losses

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 20,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        initial_actions: at.Float[at.Array, "b n ad"] | None = None,
    ) -> _model.Actions:
        observation = preprocess_observation(None, observation, train=False)
        # Note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        
        # Generate or constrain noise based on inpainting requirements
        if initial_actions is not None:
            # INPAINTING PATH: Construct constrained noise z that satisfies initial_actions
            num_initial_actions = initial_actions.shape[1]
            input_action_dim = initial_actions.shape[2]
            
            # Pad initial_actions to full model dimensions (32D) and action_horizon (30)
            if input_action_dim < self.action_dim:
                action_padding = jnp.zeros((batch_size, num_initial_actions, self.action_dim - input_action_dim))
                initial_actions_full_dim = jnp.concatenate([initial_actions, action_padding], axis=2)
            else:
                initial_actions_full_dim = initial_actions[:, :, :self.action_dim]
            
            if num_initial_actions < self.action_horizon:
                seq_padding = jnp.zeros((batch_size, self.action_horizon - num_initial_actions, self.action_dim))
                initial_actions_padded = jnp.concatenate([initial_actions_full_dim, seq_padding], axis=1)
            else:
                initial_actions_padded = initial_actions_full_dim[:, :self.action_horizon]
            
            # Compute O and U indices for inpainting (JIT-safe: static list comprehensions)
            flat_dim = self.action_horizon * self.action_dim
            
            # Build O_indices: first num_initial_actions timesteps, first input_action_dim dimensions
            O_indices = jnp.array([
                t * self.action_dim + d 
                for t in range(num_initial_actions) 
                for d in range(input_action_dim)
            ], dtype=jnp.int32)
            
            # Build U_indices: all other indices (JIT-safe: static list comprehension)
            # Python set operations happen at trace time (before JIT), so this is safe
            O_set = {t * self.action_dim + d for t in range(num_initial_actions) for d in range(input_action_dim)}
            U_indices = jnp.array([
                i for i in range(flat_dim) if i not in O_set
            ], dtype=jnp.int32)
            
            # Generate noise
            rng, noise_rng = jax.random.split(rng)
            
            if self.correlation_loaded:
                # CORRELATED NOISE: Sample with correlation matrix
                noise = self.generate_correlated_noise(noise_rng, batch_size)
            else:
                # FALLBACK: Independent noise
                noise = jax.random.normal(noise_rng, (batch_size, self.action_horizon, self.action_dim))
            
            # Extract fixed z_O and x0_O for constraint enforcement
            noise_flat = noise.reshape(batch_size, flat_dim)
            fixed_z_O = noise_flat[:, O_indices]  # [b, |O|] - fixed noise for inpainting
            x0_O = initial_actions_padded.reshape(batch_size, flat_dim)[:, O_indices]  # [b, |O|] - target actions
            
            # Precompute correction matrix for correlation-aware inpainting
            inpainting_cache = None
            if self.correlation_loaded:
                cache_key = (num_initial_actions, input_action_dim)
                if cache_key not in self.inpainting_cache:
                    logger.info(f"Computing correction matrix for {num_initial_actions} steps, {input_action_dim} dims...")
                    self.inpainting_cache[cache_key] = self._precompute_correction_matrix(O_indices, U_indices)
                inpainting_cache = self.inpainting_cache[cache_key]
            
        else:
            # NO INPAINTING: Standard noise generation
            if noise is None:
                rng, noise_rng = jax.random.split(rng)
                noise = self.generate_correlated_noise(noise_rng, batch_size)
            
            fixed_z_O = None
            x0_O = None
            O_indices = None
            inpainting_cache = None

        # Split RNG for step loop
        rng, step_rng = jax.random.split(rng)

        # Ensure FAST tokens are never used during inference
        if observation.fast_tokens is not None:
            raise ValueError(
                "FAST tokens must not be provided during inference (sample_actions). "
                "FAST tokens are only used during training for auxiliary loss. "
                "Set observation.fast_tokens=None before calling sample_actions."
            )

        # First fill KV cache with a forward pass of the prefix (no FAST tokens during inference)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        
        # Predict predicates from VLM output of base task token
        # Find base task token position (same logic as in compute_detailed_loss)
        first_predicate_token_idx = jnp.argmax(prefix_ar_mask)  # Returns index of first True
        base_task_token_idx = first_predicate_token_idx - 1
        base_task_output = prefix_out[:, base_task_token_idx, :]
        task_ids = observation.tokenized_prompt[:, 0]  # [B]
        
        # Transform KV cache for cross-layer attention
        if self.kv_transform is not None:
            kv_cache = self.kv_transform(kv_cache)
        
        def step(carry):
            x_t, time, step_rng = carry
            
            # Use config value for time threshold
            TIME_THRESHOLD_INPAINT = self.config.time_threshold_inpaint
            
            # Model forward pass
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            
            # Euler step: x_{t+dt} = x_t + dt * v_t
            x_t_new = x_t + dt * v_t
            
            # Apply correlation-aware inpainting correction
            # Only enforce when time > TIME_THRESHOLD_INPAINT (let model be free in final steps)
            if fixed_z_O is not None:
                time_new = time + dt
                
                def apply_correlated_correction(x):
                    x_flat = x.reshape(batch_size, -1)
                    
                    # Compute desired state at O: x_t[O] = (1-t)*x0[O] + t*z_O
                    x_desired_O = (1.0 - time_new) * x0_O + time_new * fixed_z_O  # [b, |O|]
                    
                    # Compute correction at O
                    delta_O = x_desired_O - x_flat[:, O_indices]  # [b, |O|]
                    
                    # Apply hard constraint at O
                    x_flat = x_flat.at[:, O_indices].set(x_desired_O)
                    
                    # If correlation matrix available, propagate correction to U
                    if inpainting_cache is not None:
                        correction_matrix = inpainting_cache['correction_matrix']  # [|U|, |O|]
                        U_indices_cached = inpainting_cache['U_indices']
                        
                        # Compute correlated correction: δ_U = Σ_{UO}Σ_{OO}^{-1} @ δ_O
                        delta_U = delta_O @ correction_matrix.T  # [b, |U|]
                        
                        # Skip if correction too large (indicates instability)
                        max_correction = jnp.max(jnp.abs(delta_U))
                        x_flat = jax.lax.cond(
                            # Prevents exploding corrections in case of noisy out of distribution initial actions
                            max_correction <= 1.0,
                            lambda x: x.at[:, U_indices_cached].add(delta_U),
                            lambda x: x,
                            x_flat
                        )
                    
                    # Sanity check: if Σ = I, correction_matrix = 0, so delta_U = 0 that is correct
                    # If the correlation is 1 everywhere we will go to the flat prediction that is correct
                    
                    return x_flat.reshape(batch_size, self.action_horizon, self.action_dim)
                
                # Only apply correction when NEW time > threshold
                x_t_new = jax.lax.cond(
                    time_new > TIME_THRESHOLD_INPAINT,
                    apply_correlated_correction,
                    lambda x: x,
                    x_t_new
                )
            
            return x_t_new, time + dt, step_rng

        def cond(carry):
            x_t, time, step_rng = carry
            # Robust to floating-point error
            return time >= -dt / 2

        x_0, _, _ = jax.lax.while_loop(cond, step, (noise, 1.0, step_rng))
        
        # Compute predicate logits for multi-label prediction
        predicate_logits = self.predicate_pred_from_vlm(base_task_output)  # [B, MAX_NUM_PREDICATES]
        # Mask invalid predicates for each task
        task_num_preds_array = jnp.array(TASK_NUM_PREDICATES, dtype=jnp.int32)
        task_num_preds = task_num_preds_array[task_ids]  # [B]
        pred_range = jnp.arange(MAX_NUM_PREDICATES)  # [20]
        valid_pred_mask = pred_range[None, :] < task_num_preds[:, None]  # [B, 20]
        predicate_logits = jnp.where(valid_pred_mask, predicate_logits, -jnp.inf)
        
        return x_0, predicate_logits
