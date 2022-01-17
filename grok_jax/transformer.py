# Adapted from the haiku transformer example at 
# https://github.com/deepmind/dm-haiku/tree/main/examples/transformer


from typing import Optional
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class CausalSelfAttention(hk.MultiHeadAttention):
  """Self attention with a causal mask applied."""

  def __call__(
      self,
      query: jnp.ndarray,
      key: Optional[jnp.ndarray] = None,
      value: Optional[jnp.ndarray] = None,
      mask: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    key = key if key is not None else query
    value = value if value is not None else query

    if query.ndim != 3:
      raise ValueError('Expect queries of shape [B, T, D].')

    seq_len = query.shape[1]
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    mask = mask * causal_mask if mask is not None else causal_mask

    return super().__call__(query, key, value, mask)


class DenseBlock(hk.Module):
  """A 2-layer MLP which widens then narrows the input."""

  def __init__(self,
               init_scale: float,
               widening_factor: int = 4,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._init_scale = init_scale
    self._widening_factor = widening_factor

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    hiddens = x.shape[-1]
    initializer = hk.initializers.VarianceScaling(self._init_scale)
    x = hk.Linear(self._widening_factor * hiddens, w_init=initializer)(x)
    x = jax.nn.gelu(x)
    return hk.Linear(hiddens, w_init=initializer)(x)


class Transformer(hk.Module):
  """A transformer stack."""

  def __init__(self,
               num_heads: int,
               num_layers: int,
               dropout_rate: float,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._dropout_rate = dropout_rate

  def __call__(self,
               h: jnp.ndarray,
               mask: Optional[jnp.ndarray],
               is_training: bool) -> jnp.ndarray:
    """Connects the transformer.

    Args:
      h: Inputs, [B, T, D].
      mask: Padding mask, [B, T].
      is_training: Whether we're training or not.

    Returns:
      Array of shape [B, T, D].
    """

    init_scale = 2. / self._num_layers
    dropout_rate = self._dropout_rate if is_training else 0.
    if mask is not None:
      mask = mask[:, None, None, :]

    # Note: names chosen to approximately match those used in the GPT-2 code;
    # see https://github.com/openai/gpt-2/blob/master/src/model.py.
    for i in range(self._num_layers):
      h_norm = layer_norm(h, name=f'h{i}_ln_1')
      h_attn = CausalSelfAttention(
          num_heads=self._num_heads,
          key_size=32,
          model_size=h.shape[-1],
          w_init_scale=init_scale,
          name=f'h{i}_attn')(h_norm, mask=mask)
      if is_training:
          h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
      h = h + h_attn
      h_norm = layer_norm(h, name=f'h{i}_ln_2')
      h_dense = DenseBlock(init_scale, name=f'h{i}_mlp')(h_norm)
      if is_training:
          h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
      h = h + h_dense
    h = layer_norm(h, name='ln_f')

    return h


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
  """Apply a unique LayerNorm to x with default settings."""
  return hk.LayerNorm(axis=-1,
                      create_scale=True,
                      create_offset=True,
                      name=name)(x)


def build_forward_fn(vocab_size: int, d_model: int, num_heads: int,
                     num_layers: int, dropout_rate: float):
    """Create the model's forward pass."""

    def forward_fn(tokens: jnp.ndarray,
                 is_training: bool = True) -> jnp.ndarray:
        """Forward pass."""
        padding_mask = jnp.greater(tokens, 0)  # masks 0 tokens (and <0, but those don't exist)
        seq_length = tokens.shape[1]

        # Embed the input tokens and positions.
        embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        token_embedding_map = hk.Embed(vocab_size, d_model, w_init=embed_init)
        token_embs = token_embedding_map(tokens)
        positional_embeddings = hk.get_parameter(
            'pos_embs', [seq_length, d_model], init=embed_init)
        input_embeddings = token_embs + positional_embeddings

        # Run the transformer over the inputs.
        transformer = Transformer(
            num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_rate)
        output_embeddings = transformer(input_embeddings, padding_mask, is_training)

        # Reverse the embeddings (untied).
        return hk.Linear(vocab_size)(output_embeddings)

    return hk.transform(forward_fn)
