import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """
    Compute self-attention by scaled dot product. 
    `query`, `key`, and `value` are computed from input token features
    using linear layers. Similarity is computed using Scaled Dot-Product
    Attention where the dot product is scaled by a factor of the square root of the
    dimension of the query vectors. See "Attention Is All You Need" for more details.

    Args for __init__:
        input_dim (int): input dimension of attention
        query_dim (int): query dimension of attention
        key_dim (int): key dimension of attention
        value_dim (int): value dimension of attention

    Inputs for forward function: 
        x (batch, num_tokens, input_dim): batch of input feature vectors for the tokens.
    Outputs from forward function:
        attn_output (batch, num_tokens, value_dim): outputs after self-attention
    """

    def __init__(self, input_dim, query_dim, key_dim, value_dim):
        super(SelfAttention, self).__init__()
        assert(query_dim == key_dim)
        self.query_dim = query_dim
        self.input_dim = input_dim
        
        # Linear layers for generating Q, K, V
        self.W_query = nn.Linear(input_dim, query_dim)
        self.W_key = nn.Linear(input_dim, key_dim)
        self.W_value = nn.Linear(input_dim, value_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        # Step 1: Compute query, key, and value matrices
        Q = self.W_query(x)  # Shape: (batch, num_tokens, query_dim)
        K = self.W_key(x)    # Shape: (batch, num_tokens, key_dim)
        V = self.W_value(x)  # Shape: (batch, num_tokens, value_dim)
        
        # Step 2: Compute the scaled dot product of Q and K.T, scaled by sqrt(query_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.query_dim, dtype=torch.float32))
        
        # Step 3: Apply softmax to obtain attention weights
        attention_weights = self.softmax(scores)
        
        # Step 4: Multiply the attention weights by the value matrix
        attn_output = torch.matmul(attention_weights, V)
        
        return attn_output

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Layer normalization module that normalizes the input over the channel dimension only.

    Args:
        input_dim (int): Dimensionality of input feature vectors.
        eps (float): Small epsilon value for numerical stability in variance calculation.

    Input:
        x (batch_size, seq_len, input_dim): Input features for tokens.

    Output:
        x_out (batch_size, seq_len, input_dim): Token features after layer normalization.
    """

    def __init__(self, input_dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        assert isinstance(input_dim, int)

        self.input_dim = input_dim
        self.eps = eps

        # Learnable parameters for scale and shift
        self.w = nn.Parameter(torch.ones(self.input_dim))  # Scale parameter initialized to 1
        self.b = nn.Parameter(torch.zeros(self.input_dim)) # Bias parameter initialized to 0

    def forward(self, x: torch.Tensor):
        assert x.shape[-1] == self.input_dim

        # Step 1: Calculate the mean along the last dimension
        mean = x.mean(dim=-1, keepdim=True)  # Shape: (batch_size, seq_len, 1)

        # Step 2: Calculate the variance along the last dimension
        variance = x.var(dim=-1, keepdim=True, unbiased=True)  # Set unbiased=True

        # Step 3: Normalize the input x
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)

        # Step 4: Apply scale and shift without reshaping w and b
        x_out = x_normalized * self.w + self.b  # Broadcasting should work here

        return x_out

        x_out = x_normalized * w + b  # Element-wise multiplication and addition

        return x_out

