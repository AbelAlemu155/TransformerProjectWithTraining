import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        
        # Store input for backward pass
        self.A = A

        Z = A @ self.W.T + self.b
        return Z

        
        # raise NotImplementedError

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass

        # Compute gradients (refer to the equations in the writeup)
        # self.dLdA = NotImplementedError
        # self.dLdW = NotImplementedError
        # self.dLdb = NotImplementedError
        # self.dLdA = NotImplementedError

        dLdA = dLdZ @ self.W

        # Flatten all batch dimensions for W and b gradients
        dLdZ_flat = dLdZ.reshape(-1, self.W.shape[0])     # (N, out_features)
        A_flat    = self.A.reshape(-1, self.W.shape[1])   # (N, in_features)

        # dW and db
        self.dLdW = dLdZ_flat.T @ A_flat                  # (out_features, in_features)
        self.dLdb = dLdZ_flat.sum(axis=0)                 # (out_features,)

        return dLdA
        
        
