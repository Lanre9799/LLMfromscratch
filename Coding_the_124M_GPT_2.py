import torch
import torch.nn as nn
import tiktoken
from Multihead_attention_class_Implementation import MultiHeadAttention
from Implementing_DummyGPT_Archi_from_scratch import batch

GPT_CONFIG_124M = {
    "vocab_size": 50257, #vocabulary size
    "context_length": 256, #Context length
    "emb_dim": 768, #Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  #Number of layers
    "drop_rate": 0.1,  #Drop rate
    "qkv_bias": False, #Query-Key-bias
}

#we will use the DummyGPT created earlier and update the dummy variables

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

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                (x + 0.044715 * torch.pow(x, 3)))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), ### Expansion
            GELU(), ###Activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), ### Contraction
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for  attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x) # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut # Add the original impact back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut #Add the original input back

        return x

# create the actual GPT model
class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Use a placeholder for LayerNorm
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self,in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# Now we initialize the 124 million parameter GPT model using the GPT_CONFIG_124M dictionary

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)


print("===============================================================> This part only pertains to parameter calculation")

# calculate the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# the reason why the total number of parameters is more than 124 million is because
# in the original GPT-2 architecture the weights from token embedding layer are reused
# in the output layer

print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)

print("===============================================================> Coding GPT-2 to predict the next token")

### code to predict the next GPT-2 token
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    # The line for _ in range(max_new_tokens) below is for the number of
    # iterations to generate the number
    # of tokens needed
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and context size is 10
        # then only 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        # takes the last row required for making the next prediction
        logits = logits[:, -1, :]

        #Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1) # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=-1) # (batch, n_tokens+1)

    return idx

# let's now try to generate a text sample
# we first encode the input context into token IDs


tokenizer = tiktoken.get_encoding("gpt2") # Make an instance of the tokenizer
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded :",encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape :",encoded_tensor.shape)

print("===============================================================> Evaluating the model")

# putting the model into .eval() mode, disables random components like dropout
model.eval() #A
out = generate_text_simple(
model=model,
idx=encoded_tensor,
max_new_tokens=6, # number of new words you want generated
context_size=GPT_CONFIG_124M["context_length"]
)
print("Output :",out)
print("Output length :", len(out[0]))

# Decode the predicted tokens to text using .decode

decode_text = tokenizer.decode(out.squeeze(0).tolist())
print(decode_text)

# The reason why the output  text is random, is because the parameters have not been
# trained yet