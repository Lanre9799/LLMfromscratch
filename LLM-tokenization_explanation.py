# re used to split text

import re
text = "Hello, world. Is this-- a test."
# re.split splits words at white spaces
# \s splits white spaces
# [,.] splits , and .
result = re.split(r'([,.:;?_!"()]|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

#removes white spaces from text
result = [item for item in result if item.strip()]
print(result)
# read the verdict file as raw_text
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# print the number of characters
print("Total number of characters: ", len(raw_text))
#print the first 100 words
print(raw_text[:99])

# The goal is to tokenize the 20479 tokens and turn them to token IDS
# Now we tokenize the raw_text with the basic tokenizer created above

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])

print(len(preprocessed))

# The verdict has now been tokenized, now be convert to token IDS
# we use set to get the unique tokens
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

# After getting vocabulary size, we'll now be making the vocabulary dictionary
vocab = {token:integer for integer, token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

# create a tokenizer class that contains encode method and decode method
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# instantiate a new tokenizer object from simpletokenizerV1 class and tokenize a text

tokenizer = SimpleTokenizerV1(vocab)
# the words used in text are present in the vocabulary and thus the decoding
# of the sentence is possible
text = """"It's the last he painted, you know,"
            Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

# Now we decode the token IDs
text = tokenizer.decode(ids)
print(text)

# If word is not present in the vocabulary we'll have an error during encoding
# The error can be avoided by addd=ing special context tokens
# Adding special context tokens
# we'll modify the tokenizer to handle missing words in the vocabulary
# We'll do this by implementing SimpleTokenizerV2
# <|unk|> - unknown
# and <|endoftext|>

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])

vocab = {token:integer for integer, token in enumerate(all_tokens)}

print(len(vocab.items()))

# we have now added UNK and endoftext to the vocabulary hence the change in length to 1132
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        #if item is in not in the preprocessed data return unk
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text)

ids = tokenizer.encode(text)
print(ids)

text = tokenizer.decode(ids)
print(text)

# along with these special tokens there are other tokens which researchers also consider
# [BOS] (beginning of sequence): This token marks the start of a text
# [EOS] (end of sequence): This token is positioned at the end of a text, and is especially useful when concatenating multiple unrelated texts
# [PAD] (padding): When training LLMS with batch sizes larger than one, the batch might contain texts of varying lengths. To ensure all text have the same length
# Shorter texts are extended or "padded" using the [PAD] token
# GPT does not use the tokens mentioned above, they only use endoftext for simplicity
# GPT does not use UNK for unknown tokens or out of vocabulary words.
# Instead GPT uses byte pair encoding tokenizer which breaks down word into subword units
