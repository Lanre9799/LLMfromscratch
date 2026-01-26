# Datasets and DATALOADER is used as a prerequisite for positional embedding
import torch
from Dataset_and_Dataloader import create_dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_txt = f.read()


print("PyTorch version: ", torch.__version__)
dataloader = create_dataloader(raw_txt, batch_size=8, max_length=4, stride=4, shuffle=False)

date_iter = iter(dataloader)
#first_batch = next(date_iter)
#print(first_batch)
inputs, targets = next(date_iter)
print("Inputs:\n", inputs)
print("\nShape:\n", inputs.shape)


#second_batch = next(date_iter)
#print(second_batch)


#embedding layer for token embedding
vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Now let's do the embedding for our dataset

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# create the embedding layer for absolute position embedding approach

max_length= 4
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)

# token embedding + Position embedding = input embedding
# we calculate input embedding

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)




