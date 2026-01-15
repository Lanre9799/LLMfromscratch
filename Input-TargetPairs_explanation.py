import importlib
import tiktoken


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

#intantiate tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

#take a sample of the first 50 tokens for demonstration
enc_sample = enc_text[50:]

# The most intuitive ways to create the input-target pairs for the next word
# prediction task is to create two variables x and y, the x variable is the
# input and the y variable is the target
# context size determines how many tokens are included in the input
context_size = 4 #length of the input
# The context_size of 4 means that the model is trained to look at a sequence of 4 words (or tokens)
# To predict the next word in the sequence.
# The input x is the first 4 tokens [1,2,3,4], and the target y is the next 4 tokens [2,3,4,5]

x= enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")

# for next word prediction task, processing the inputs along with the targets
#which are inputs shifted by one

for i in range(1,context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(context, "------>", desired)

for i in range(1,context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(tokenizer.decode(context), "------>", tokenizer.decode([desired]))

