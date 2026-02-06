import torch
from Measuring_LLM_Loss_Function import tokenizer
from Dataset_and_Dataloader import create_dataloader
from Coding_the_124M_GPT_2 import model, GPT_CONFIG_124M


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    text_data = f.read()

# print the number of characters
print("Total number of characters: ", len(text_data))

#print the first 100 words
print(text_data[:99])

#print the last 100 characters
print(text_data[-99:])

#find the total number characters
total_characters = len(text_data)

#tokenize and find the number of tokens
total_tokens = len(tokenizer.encode(text_data))

print("characters:", total_characters)
print("Tokens:", total_tokens)

# Create the training and validation ratio
train_ratio = 0.90
split_idx = int(train_ratio * total_characters)
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


torch.manual_seed(123)

train_loader = create_dataloader(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)

val_loader = create_dataloader(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)

print("Train loader: ")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader: ")
for x, y in val_loader:
    print(x.shape, y.shape)

print(len(train_loader))

# print out the training and validation tokens
train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("Train tokens: ", train_tokens)
print("Validation tokens: ", val_tokens)
print("All tokens: ", train_tokens + val_tokens)


#calculate cross entropy loss

#loss for one batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

#Loss for multiple batches
def calc_loss_loader(data_loader, model, device, num_batches):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the dataloader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note:
# Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
# which is approximately 2x faster than on an apple CPU (as measured on an M3 MacBook Air)
# However, the resulting loss values may be slightly different.

#if torch.cuda.is_available():
#     device = torch.device("cuda")
#elif torch.backends.mps.is_available():
#     device = torch.device("mps")
#else:
#     device = torch.device("cpu")
#
#  print(f"Using {device} device.")

model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes

torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=None)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=None)

print("Training loss: ", train_loss)
print("Validation loss: ", val_loss)