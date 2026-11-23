from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention:
    """
    Multi Head Attention
    """
    def __init__(self, embed_dim, num_heads):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Initialize parameters and layers
        # DO NOT MODIFY
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Initialize scaled dot product attention
        self.attention = ScaledDotProductAttention()

        # Linear layers: project to embed_dim and back
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        """
        Initialize the weights and biases with the given values.
        """
        # Initialize your linear layers (DO NOT MODIFY)
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where 1/True indicates positions to ignore
        :param attn_mask: (L, S) where 1/True indicates positions to ignore
        :return: (N, L, E)
        """

        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]
        d_k = self.embed_dim // self.num_heads

        # linear projections
        q = self.q_proj.forward(query)  # (N, L, embed_dim)
        k = self.k_proj.forward(key)    # (N, S, embed_dim)
        v = self.v_proj.forward(value)  # (N, S, embed_dim)

        # split heads
        q = self._split_heads(q)  # (N, H, L, d_k)
        k = self._split_heads(k)  # (N, H, S, d_k)
        v = self._split_heads(v)  # (N, H, S, d_k)

        # merge masks
        mask = self._merge_masks(key_padding_mask, attn_mask)  # (N, H, L, S) or None

        # Apply attention per head
        attn_outputs = self.attention.forward(q, k, v, mask=mask)  # (N, H, L, d_k)

        # concat heads
        attn_output = self._concat_heads(attn_outputs)  # (N, L, embed_dim)

        # final linear projection
        output = self.out_proj.forward(attn_output)  # (N, L, embed_dim)

        # store useful tensors for backward
        self.q = q; self.k = k; self.v = v
        self.attn_outputs = attn_outputs
        self.attn_output = attn_output
        self.mask = mask

        return output

    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, L, E)
        :return: Gradient of loss wrt input query, key, value of shapes (N, L, E), (N, S, E), (N, S, E)
        """

        # backprop through final linear
        d_attn_output = self.out_proj.backward(d_output)  # (N, L, embed_dim)

        # split grads into heads shape
        d_attn_outputs = self._split_heads(d_attn_output)  # (N, H, L, d_k)

        # backprop through attention
        d_q_heads, d_k_heads, d_v_heads = self.attention.backward(d_attn_outputs)  # each shapes: (N,H,L,d_k) or (N,H,S,d_k)

        # concat heads back to (N, L, embed_dim) and (N, S, embed_dim)
        d_q = self._concat_heads(d_q_heads)  # (N, L, embed_dim)
        d_k = self._concat_heads(d_k_heads)  # (N, S, embed_dim)
        d_v = self._concat_heads(d_v_heads)  # (N, S, embed_dim)

        # backprop through input projections
        d_query = self.q_proj.backward(d_q)  # (N, L, E)
        d_key = self.k_proj.backward(d_k)    # (N, S, E)
        d_value = self.v_proj.backward(d_v)  # (N, S, E)

        return d_query, d_key, d_value

    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        Merge key_padding_mask and attn_mask into a single mask.
        :param key_padding_mask: (N, S)
        :param attn_mask: (L, S)
        :return: (N, H, L, S) or None
        """
        if key_padding_mask is None and attn_mask is None:
            return None

        N = self.N
        H = self.num_heads
        L = self.L
        S = self.S

        if key_padding_mask is None:
            key_mask = np.zeros((N, 1, 1, S), dtype=bool)
        else:
            # key_padding_mask: (N, S) -> (N, 1, 1, S)
            key_mask = key_padding_mask.reshape(N, 1, 1, S).astype(bool)

        if attn_mask is None:
            attn_m = np.zeros((1, 1, L, S), dtype=bool)
        else:
            attn_m = attn_mask.reshape(1, 1, L, S).astype(bool)

        # broadcast to (N, H, L, S)
        key_mask = np.broadcast_to(key_mask, (N, H, L, S))
        attn_m = np.broadcast_to(attn_m, (N, H, L, S))

        combined_mask = np.logical_or(key_mask, attn_m)
        return combined_mask

    def _split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the front.
        :param x: (N, L, embed_dim) or (N, L, embed_dim) for gradient as well
        :return: (N, num_heads, L, embed_dim // num_heads)
        """
        N, L, _ = x.shape
        d_k = self.embed_dim // self.num_heads
        # Reshape then transpose
        x = x.reshape(N, L, self.num_heads, d_k)  # (N, L, H, d_k)
        x = np.transpose(x, (0, 2, 1, 3))         # (N, H, L, d_k)
        return x

    def _concat_heads(self, x):
        """
        Concatenate the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the back.
        :param x: (N, num_heads, L, embed_dim // num_heads)
        :return: (N, L, embed_dim)
        """
        # transpose to (N, L, H, d_k)
        x = np.transpose(x, (0, 2, 1, 3))
        N, L, H, d_k = x.shape
        x = x.reshape(N, L, H * d_k)  # (N, L, embed_dim)
        return x
