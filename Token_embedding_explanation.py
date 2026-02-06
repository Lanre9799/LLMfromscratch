# we're starting with a toy token embedding for an example
# word2vec-google-news-300
#300 dimension word2Vec
import torch
import gensim.downloader as api
model = api.load("word2vec-google-news-300") # download the model and return as object ready for use

word_vectors = model

# Let us look how the vector embedding of a word looks like
print(word_vectors['computer']) #Example: Accessing the vector for the word 'computer'

print(word_vectors['cat'].shape)

#King + Woman - Man =?

# Example of using most_similar
print(word_vectors.most_similar(positive=['king','woman'], negative=['man'], topn=10 ))

#Example of calculating similarity
print(word_vectors.similarity('woman', 'man'))
print(word_vectors.similarity('king', 'queen'))
print(word_vectors.similarity('uncle', 'aunt'))
print(word_vectors.similarity('boy', 'girl'))
print(word_vectors.similarity('nephew', 'niece'))
print(word_vectors.similarity('paper', 'water'))

#Most similar words

print(word_vectors.most_similar("tower", topn=5))

import numpy as np
# words to compare
word1 = 'man'
word2 = 'woman'

word3 = 'semiconductor'
word4 = 'earthworm'

word5 = 'nephew'
word6 = 'niece'

#calculate the vector difference
vector_difference1 = model[word1] - model[word2]
vector_difference2 = model[word3] - model[word4]
vector_difference3 = model[word5] - model[word6]

#calculate the magnitude of the vector difference
magnitude_of_difference1 = np.linalg.norm(vector_difference1)
magnitude_of_difference2 = np.linalg.norm(vector_difference2)
magnitude_of_difference3 = np.linalg.norm(vector_difference3)

#print the magnitude of the difference
print("The magnitude of the difference between '{}' and '{}' is {:.2f}".format(word1, word2, magnitude_of_difference1))
print("The magnitude of the difference between '{}' and '{}' is {:.2f}".format(word3, word4, magnitude_of_difference2))
print("The magnitude of the difference between '{}' and '{}' is {:.2f}".format(word5, word6, magnitude_of_difference3))

# magnitude of difference distance is an indication of how close or how far words are in their meaning

# Creating Token Embeddings

input_ids = torch.tensor([2,3,5,1])

#we're using a small vocabulary of six words and  assuming the vector dimension is 3
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
# torch.nn.Embedding simple lookup table that stores embedding of a fixed dictionary and size
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight)
# these embedding layer weight will still be optimized

# To get the embedding vector of a particular token ID, embedding layer is a lookup matrix
print(embedding_layer(torch.tensor([3])))

# get the embedding vector of input ids stated above
print(embedding_layer(input_ids))





