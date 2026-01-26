import torch
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], #Your (X^1)
     [0.55, 0.87, 0.66], #journey (X^2)
     [0.57, 0.85, 0.64], # starts (X^3)
     [0.22, 0.58, 0.33], # with (X^4)
     [0.77, 0.25, 0.10], # one (X^5)
     [0.05, 0.80, 0.55]] # step (X^6)
)

# corresponding words
words = ['Your', 'journey', 'starts', 'with', 'one', 'step']

#Extract x,y,z coordinates
x_coords = inputs[:, 0].numpy()
y_coords = inputs[:, 1].numpy()
z_coords = inputs[:, 2].numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each point and annotate with corresponding word
for x, y, z, word in zip(x_coords, y_coords, z_coords, words):
 ax.scatter(x, y, z)
 ax.text(x, y, z, word, fontsize=10)

 # Set labels for axes
 ax.set_xlabel('X')
 ax.set_ylabel('Y')
 ax.set_zlabel('Z')

 # Create 3D plot with vectors from origin to each point, using different colors
 fig = plt.figure()
 ax = fig.add_subplot(111, projection='3d')

 # Define a list of colors for the vectors
 colors = ['r', 'g', 'b', 'c', 'm', 'y']

 # Plot each vector with a different color and annotate with the corresponding word
 for (x, y, z, word, color) in zip(x_coords, y_coords, z_coords, words, colors):
  #Draw vector from origin to the point (x,y,z) with specified color and smaller arrows
  ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.05)
  ax.text(x, y, z, word, fontsize=10, color=color)

 #set labels for axes
 ax.set_xlabel('X')
 ax.set_ylabel('Y')
 ax.set_zlabel('Z')

 # Set plot limits to keep arrows within the plot boundaries
 ax.set_xlim([0, 1])
 ax.set_ylim([0, 1])
 ax.set_zlim([0, 1])

 plt.title('3D Plot of Word Embeddings with colored Vectors')
 plt.show()

 query = inputs[1] # 2nd input token is the query

# calculate the attention score of the vector embedding relative to the query token
# this is done with the dot product
# vector embedding dot product query
 attn_scores_2 = torch.empty(inputs.shape[0])
 for i, x_i in enumerate(inputs):
  attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not needed)

print(attn_scores_2)

# Normalize the attention score so it can be more interpretable
# Make the attention scores sum up to 1.....100%
# This is attention weight
attn_weights_2_tmp = attn_scores_2/ attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# It's better to normalize with softmax
# create softmax function to normalize attention score and get attention weight

def softmax_naive(x):
  return torch.exp(x) / torch.exp(x).sum(dim=0)
attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

#getting attention weights with softmax function in PyTorch
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

#calculating context vector
query = inputs[1] # 2nd input token is the query

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
 context_vec_2 += attn_weights_2[i] * x_i

print(context_vec_2)

# To get the attention scores of the whole input embedding
# we'll create a loop inside a loop this allows all the attention score
#for all queries to be generated

attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
 for j, x_j in enumerate(inputs):
  attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

# this method of getting context vector is computationally expensive
# this same result can be achieved with vector multiplication:
# multiplying input matrix by input transpose

attn_scores = inputs @ inputs.T
print(attn_scores)

# use softmax to find the attention weights by normalizing attention scores

attn_weights = torch.softmax(attn_scores, dim=-1) # dim parameter specifies the dimension of the input tensor along which the function will be computed
print(attn_weights)

# By implementing a dim of -1 we are instructing the softmax function to apply the normalization along the last dimension of the attn_scores tensor
# dim = -1 helps in normalizing across the column, so each row sums up to 1

#Now we calculate the context vector by multiplying vector embedding with corresponding attention weight

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)