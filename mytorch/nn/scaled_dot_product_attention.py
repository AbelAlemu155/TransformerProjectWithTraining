import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        self.eps = 1e10  # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)   # softmax along S dimension
    
    def forward(self, Q, K, V, mask=None):
        # Store for backward
        self.Q = Q
        self.K = K
        self.V = V

        E = Q.shape[-1]                     # key/query dimension

        # (N,...,H,L,E) @ (N,...,H,E,S) → (N,...,H,L,S)
        KT = np.swapaxes(K, -2, -1)
        scaled_dot_product = Q @ KT / np.sqrt(E)

        # Apply mask
        if mask is not None:
            scaled_dot_product = scaled_dot_product - mask * self.eps

        # Apply softmax
        self.attention_scores = self.softmax.forward(scaled_dot_product)

        # Output: (N,...,H,L,S) @ (N,...,H,S,Ev) → (N,...,H,L,Ev)
        output = self.attention_scores @ V

        return output
    
    def backward(self, d_output):
        Q, K, V = self.Q, self.K, self.V
        A = self.attention_scores

        E = Q.shape[-1]

        # -------- dV --------
        # (N,...,H,S,L) @ (N,...,H,L,Ev)
        AT = np.swapaxes(A, -2, -1)
        d_V = AT @ d_output

        # -------- dA (before softmax backward) --------
        VT = np.swapaxes(V, -2, -1)
        d_attention_scores = d_output @ VT

        # -------- dZ through softmax --------
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)

        # scale gradient
        d_scaled_dot_product /= np.sqrt(E)

        # -------- dQ --------
        KT = np.swapaxes(K, -2, -1)
        d_Q = d_scaled_dot_product @ K

        # -------- dK --------
        QT = np.swapaxes(Q, -2, -1)
        d_K = np.swapaxes(d_scaled_dot_product, -2, -1) @ Q

        return d_Q, d_K, d_V
