import torch
import numpy as np
import math

from torch import nn

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], #Your (X^1)
     [0.55, 0.87, 0.66], #journey (X^2)
     [0.57, 0.85, 0.64], # starts (X^3)
     [0.22, 0.58, 0.33], # with (X^4)
     [0.77, 0.25, 0.10], # one (X^5)
     [0.05, 0.80, 0.55]] # step (X^6)
)
# A the second input element
# B the input embedding size, d=3
# C the output embedding size d_out
x_2 = inputs[1] #A
d_in = inputs.shape[1] #B
d_out = 2 #C

# Note that in GPT-like models the input and output dimensions are usually the same.
#initialize the three weight matrices Wq,Wk and Wv
# require-grad set to False. This reduces clutter, however during backpropagation when training
# it's set to TRUE

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)

print(W_query)
print(W_key)
print(W_value)

# Get the Query,Keys, Values for x_2

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print("Queries", query_2)
print("Keys", key_2)
print("Values", value_2)

#Get the Query,Keys and Values for the input embedding

keys = inputs @ W_key
values = inputs @ W_value
queries = inputs @ W_query
print("Queries shape", queries.shape)
print("Keys shape", keys.shape)
print("Values shape", values.shape)

# The result is the successful projection of 3 dimensional vector into 2 dimensional space
# attention scores is the dot product of keys and queries

# attention score for journey in context of journey
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# attention score for journey in context of the whole sequence
attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print(attn_scores_2)

# attention score for the whole dataset
attn_scores = queries @ keys.T #omega
print(attn_scores)

# scale by SQRT(d-keys) before normalization, the dimension of keys is 2

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)
print(d_k)

#Why Divide by SQRT (Dimension)
#1. For stability in learning

tensor = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])

#Apply softmax without scaling
softmax_result = torch.softmax(tensor, dim=-1)
print("Softmax without scaling: ", softmax_result)

#Multiply the tensor by 8 and then apply softmax
scaled_tensor = tensor * 8
softmax_scaled_result = torch.softmax(scaled_tensor, dim=-1)
print("Softmax without scaling (tensor * 8): ", softmax_scaled_result)

#2. SQRT is related to the variance. To make the variance of the dot product stable

# Function to compute variance before and after scaling
def compute_variance(dim, num_trials=1000):
    dot_products = []
    scaled_dot_products = []

    # Generate multiple random vectors and compute dot products
    for _ in range(num_trials):
        q = np.random.randn(dim)
        k = np.random.randn(dim)

        # Compute dot product
        dot_product = np.dot(q, k)
        dot_products.append(dot_product)

        # Scale the dot product by sqrt(dim)
        scaled_dot_product = dot_product / np.sqrt(dim)
        scaled_dot_products.append(scaled_dot_product)

    # Calculate variance of the dot products
    variance_before_scaling = np.var(dot_products)
    variance_after_scaling = np.var(scaled_dot_products)

    return variance_before_scaling, variance_after_scaling

# For dimension 5
variance_before_5, variance_after_5 = compute_variance(5)
print(f"Variance before scaling (dim=5): {variance_before_5}")
print(f"Variance after scaling (dim=5): {variance_after_5}")

# For dimension 20
variance_before_20, variance_after_20 = compute_variance(20)
print(f"Variance before scaling (dim=20): {variance_before_20}")
print(f"Variance after scaling (dim=20): {variance_after_20}")

# compute context vector attention weight * keys

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out): # takes the input dimension and output dimension
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        query = x @ self.W_query
        key = x @ self.W_key
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

# We can improve SelfAttention_v1 implementation further by utilizing Pytorch's nn.linear layers
# using nn.linear instead of nn.parameter(torch.rand((...)), is advantageous because nn.Linear has an optimized weight initialization scheme,
# contributing to more stable and effective model training
# optimized version

class SelfAttention_v2(nn.Module):

   def __init__(self, d_in, d_out):
       super().__init__()
       self.W_query = nn.Linear(d_in, d_out, bias=False)
       self.W_key = nn.Linear(d_in, d_out, bias=False)
       self.W_value = nn.Linear(d_in, d_out, bias=False)

   def forward(self, x):
       key = self.W_key(x)
       query = self.W_query(x)
       value = self.W_value(x)

       attn_scores = query @ keys.T
       attn_weights = torch.softmax(attn_scores / keys.shape[-1], dim=-1)

       context_vec = attn_weights @ values
       return context_vec

# You can use SelfAttention_v2 similar to SelfAttention_v1

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

##Causal attention

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

# generate a mask with the values above diagonal as 0
# using lower triangular matrix tril

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

#masked attention score
# attention weight x masked matrix

masked_simple = attn_weights * mask_simple
print(masked_simple)

# renormalize the attention weights to sum up ti 1 again in each row

row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

# This method of calculating the masked attention weight
# for context vector calculation is inefficient as it leads to data leakage

# A more efficient method for calculating attention weight
print(attn_scores)

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

# Attention weight calculation
attn_weights = torch.softmax(masked/ keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

# dropout prevents overfitting
# Masking additional attention weights with dropout
# we use a dropout rate of 50%, which means masking out half of the attention weights.
# we use lower dropout rate, such as 0.1 or 0.2
example = torch.ones(6,6)
print(example)

#dropout turns off 50% of the neurons
# and scales the other values that are not turned off
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) #A
print(dropout(example))

#apply dropout to attention weight
#drop out is probabilistic so it zeros out a random 50% of the attention weight
torch.manual_seed(123)
print(dropout(attn_weights))

# Implementing a causal attention class with dropout modifications
# ensure it can handle batches consisting of more than one input
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

# causal attention class
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) #buffers are moved to the appropriate device, it's relevant for future processes

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2) # changed transpose
        attn_scores.masked_fill( # New, _ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # :num_tokens to account for cases
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vec = ca(batch)
print("context_vec.shape:", context_vec.shape)
print(context_vec)

#Extending single head attention to multi-head attention
#Multi-head attention refers to dividing attention mechanism to multiple heads
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

# get the parameters needed for clarity and understanding
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], #Your (X^1)
     [0.55, 0.87, 0.66], #journey (X^2)
     [0.57, 0.85, 0.64], # starts (X^3)
     [0.22, 0.58, 0.33], # with (X^4)
     [0.77, 0.25, 0.10], # one (X^5)
     [0.05, 0.80, 0.55]] # step (X^6)
)

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

#Get the multihead attention
torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape: ",context_vecs.shape)


