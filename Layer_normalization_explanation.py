import torch
import torch.nn as nn

torch.manual_seed(123)
batch_example = torch.randn(2, 5) #A
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
output = layer(batch_example)
print(output)

# We're going to apply the layer normalization to this simple neural network
# dim = -1 means along the column for both mean and variance
# keep dim means keep the dimension the same
mean = output.mean(dim=-1, keepdim=True)
var = output.var(dim=-1, keepdim=True)
print("Mean:\n",mean)
print("Variance:\n",var)

#find the normalized layer output
out_norm = (output - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer output:\n",out_norm)
print("Mean:\n",mean)
print("Variance:\n",var)

# To improve readability we turn off scientific notation, we approximates out mean to 0

torch.set_printoptions(sci_mode=False)
print("Mean:\n",mean)
print("Variance:\n",var)

# Creating layer Normalization class

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True, unbiased=False)
print("Mean:\n",mean)
print("Variance:\n",var)