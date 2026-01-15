# Datasets and DATALOADER
# Iterates over the input dataset and returns the input and targets as Pytorch tensors
# we need the data in tensors because for upcoming optimization processes we are
# working with pytorch which operates with tensors(multidimensional array)
# GPTDatasetV1 defines how individual rows are fetched from the dataset
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1: i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
# get item tells the dataloader the kind of input and target we should have
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# step 1: Initialize the tokenizer
# step 2: Create dataset
# step 3: drop_last=True drops the last batch if it is shorter than the specified batch_size to prevent loss spikes during training
# step 4: The number of CPU processes to use for preprocessing

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # batch_size the number of processes program runs to update parameter
    # num_worker for parallel processing on different threads of the CPU
    #initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    #create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    #create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last = drop_last, num_workers=num_workers)
    return dataloader
# we test the dataloader with a batch size of 1 for an LLM with context size 4

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_txt = f.read()

print("PyTorch version: ", torch.__version__)
dataloader = create_dataloader(raw_txt, batch_size=8, max_length=4, stride=4, shuffle=False)

date_iter = iter(dataloader)
#first_batch = next(date_iter)
#print(first_batch)
inputs, targets = next(date_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)


#second_batch = next(date_iter)
#print(second_batch)

