import torch
import time
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
from Dataset_and_Dataloader import create_dataloader
from Coding_the_124M_GPT_2 import model, GPT_CONFIG_124M, generate_text_simple, GPTModel
from Measuring_LLM_Loss_Function import tokenizer, text_to_token_ids, token_ids_to_text

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
    drop_last=False,
    shuffle=False,
    num_workers=0,
)

# Sanity check

if total_tokens * train_ratio < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the training loader."
          "Try to lower the GPT_CONFIG_124M[ context length] or."
          "increase the training_ratio")

if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the validation loader."
          "Try to lower the GPT_CONFIG_124M[ context length] or."
          "increase the training_ratio")

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
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
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
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("Training loss: ", train_loss)
print("Validation loss: ", val_loss)


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    #main training loop
    for epoch in range(num_epochs):
        model.train()  #set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel() # Returns the total number of elements (or tokens) in the input_batch
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep{epoch+1} (Step{global_step:06d}): "
                      f"Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f} ")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model,tokenizer,device, start_context
        )

# This evaluate_model code calculates the loss over the training and validation set while ensuring that the model is in eval mode
# so that dropout is disabled when calculating the loss over the training and validation sets
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# This function lets us generate the text after every epoch
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n ", "")) # Compact print format
    model.train()


# Let's see all this in action by training a GPTModel instance for 10 epochs, using AdamW optimizer


start_time = time.time()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10

train_model_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# Note:
# Uncomment the following code to show the execution time
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
