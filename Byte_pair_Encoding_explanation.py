# tiktoken is a byte pair encoder used by openAI

import importlib
import tiktoken

#intantiate tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

#The last text in the tokenizer vocabulary is the endoftext,
#this is the same tokenizer used for GPT and shows the last token ID in the vocabulary is 50256

integers = tokenizer.encode("Akwirw ier")
print(integers)

strings = tokenizer.decode(integers)
print(strings)