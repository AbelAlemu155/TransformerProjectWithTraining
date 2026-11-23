import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, Z):
        """
        Z: (..., C)
        """
        # Normalize dim index
        dim = self.dim if self.dim >= 0 else Z.ndim + self.dim

        # For numerical stability
        Z_max = np.max(Z, axis=dim, keepdims=True)
        exp_Z = np.exp(Z - Z_max)

        A = exp_Z / np.sum(exp_Z, axis=dim, keepdims=True)

        self.A = A
        return A

    def backward(self, dLdA):
        """
        dLdA: same shape as A
        """
        A = self.A
        dim = self.dim if self.dim >= 0 else A.ndim + self.dim

        # The gradient formula:
        # dZ = A * (dA - sum(dA * A)) along softmax dim
        dot = np.sum(dLdA * A, axis=dim, keepdims=True)

        dLdZ = A * (dLdA - dot)
        return dLdZ
