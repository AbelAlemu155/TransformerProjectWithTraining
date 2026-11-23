import torch
from torch import nn
import math

# # '''
# # TODO: Implement this Module.

# # Specification:
# # - Module should add positional information to input embeddings
# # - Uses sinusoidal position encodings as described in "Attention Is All You Need"
# # - Positional encoding matrix should have shape (1, max_len, d_model)
# # - Even indices use sine functions, odd indices use cosine functions
# # - Wavelengths form geometric progression from 2π to 10000·2π
# # - Encoding values should be on same device as input tensor
# # - Should handle any sequence length up to max_len
# # - Should raise error if input sequence length exceeds max_len
# # '''
# # class PositionalEncoding(nn.Module):
# #     def __init__(self, d_model, max_len):
# #         """
# #         Initialize the PositionalEncoding.
# #         Args:
# #             d_model (int): The dimension of the model.
# #             max_len (int): The maximum length of the input sequence.
        
# #         Steps:
# #         1. Call parent class constructor using super().__init__()
# #         2. Call create_pe_table to initialize positional encoding matrix
# #         """     
# #         super().__init__()
# #         self.create_pe_table(d_model, max_len)

# #     def create_pe_table(self, d_model, max_len):
# #         """
# #         Create the positional encoding table.
        
# #         Args:
# #             d_model (int): The dimension of the model.
# #             max_len (int): The maximum length of the input sequence.
        
# #         Side Effects:
# #             - Initializes the positional encoding buffer 'pe' 
# #               of shape (1, max_len, d_model) (in order to broadcast with input tensor)
# #         """
# #         # TODO: Implement create_pe_table
# #         raise NotImplementedError # Remove once implemented
# #         pe = NotImplementedError
# #         # Register as buffer to save with model state
# #         self.register_buffer('pe', pe)
        

# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         """
# #         Forward pass for the PositionalEncoding.
# #         Args:
# #             x (torch.Tensor): The input tensor of shape (B x T x d_model)
# #         Returns:
# #             torch.Tensor: Input with positional encoding added (B x T x d_model)
# #         Errors:
# #             - ValueError: If sequence length exceeds maximum length 
# #         """
# #         # TODO: Implement forward
# #         # Step 1: Get sequence length from input tensor
# #         seq_len = x.size(1)
# #         # Step 2: Verify sequence length doesn't exceed maximum length, raise error if it does
# #         if seq_len > self.pe.size(1):
# #             raise ValueError(f"Sequence length {seq_len} exceeds the maximum length {self.pe.size(1)}")
# #         # Step 3: Add positional encodings to input
# #         raise NotImplementedError # Remove once implemented


# import torch
# import torch.nn as nn
# import math

# class PositionalEncoding(nn.Module):
#     """
#     Adds sinusoidal positional encoding to input embeddings.
#     """
#     def __init__(self, d_model: int, max_len: int):
#         """
#         Args:
#             d_model: Embedding dimension
#             max_len: Maximum sequence length
#         """
#         super().__init__()
#         self.create_pe_table(d_model, max_len)

#     def create_pe_table(self, d_model: int, max_len: int):
#         """
#         Creates positional encoding table (1, max_len, d_model)
#         """
#         # Initialize positional encoding matrix
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # shape: (max_len, 1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2, dtype=torch.float32) * 
#             (-math.log(10000.0) / d_model)
#         )  # geometric progression

#         # Apply sin to even indices, cos to odd indices
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         # Shape to (1, max_len, d_model) to broadcast with input
#         pe = pe.unsqueeze(0)  # (1, max_len, d_model)

#         # Register as buffer so it's saved with model but not a parameter
#         self.register_buffer('pe', pe)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: Input tensor (B x T x d_model)
#         Returns:
#             x + positional encoding (B x T x d_model)
#         """
#         seq_len = x.size(1)
#         if seq_len > self.pe.size(1):
#             raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.pe.size(1)}")

#         # Add positional encoding to input (broadcast along batch dimension)
#         return x + self.pe[:, :seq_len, :].to(x.device)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in "Attention Is All You Need".
    """
    def __init__(self, d_model: int, max_len: int):
        """
        Args:
            d_model: model dimension
            max_len: maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.create_pe_table(d_model, max_len)

    def create_pe_table(self, d_model, max_len):
        """
        Create the positional encoding matrix (1, max_len, d_model)
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")
        return x + self.pe[:, :seq_len, :].to(x.device)
