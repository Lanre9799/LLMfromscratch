import tiktoken
import torch
from Coding_the_124M_GPT_2 import model, GPT_CONFIG_124M, generate_text_simple



def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())


start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx = text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
print("===============================================================> Calculating the text generation loss: Cross-entropy and perplexity")
# input-target pairs
inputs = torch.tensor([[16833, 3626, 6100], #["every effort moves",
                       [40, 1107, 588]]) # "I really like" ]

targets = torch.tensor([[3626, 6100, 345], # ["effort moves you",
                        [1107, 588, 11311]]) # "really like chocolate"]

# Applying softmax function, we can turn the logits tensor into a tensor of the same dimension

with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary being generated next
print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDS:\n", token_ids)


#comparing our output with target
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

# The token probabilities corresponding to the target indices are as follows
# this allows us to obtain the targets probabilities

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

# Merge the targets probabilities
# compute the logarithm of all token probabilities
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print("Log probas:\n", log_probas)

#Calculate the average probability for each token
avg_log_probas = torch.mean(log_probas)
print("Average log probas:\n", avg_log_probas)

# The goal is to make this average log probability as large as possible by optimizing the
# model weights. Due to log, the largest possible value is 0, and we are currently far
# away from 0. In deep learning, instead of maximizing the average log-probability, it's
# standard convention to minimize it. So instead of maximizing -10.7722 so that it approaches 0
# in deep learning, we would minimize 10.7722. This value is called cross-entropy loss

neg_avg_log_probas = avg_log_probas * -1
print("Negative log probas:\n", neg_avg_log_probas)

# Logits has shape ( batch_size, num_tokens, vocab_size)
print("Logits:\n", logits)

# Targets have shape (batch_size, num_tokens)
print("Targets shape: ", targets.shape)

# For the cross_entropy function in Pytorch, we want to flatten these tensors by combining them over num_tokens

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()

print("Flattened logits:\n", logits_flat.shape)
print("Flattened targets:\n", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print("Loss:", loss)

# Perplexity is another intuitive way of calculating loss
# It is more understandable than loss as it provides a number that can be directly
# related with the vocab_size

perplexity = torch.exp(loss)
print("Perplexity:", perplexity)