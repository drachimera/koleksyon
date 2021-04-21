import torch
import numpy as np


# ======================================================================= #
#                      Tensor Indexing
# ======================================================================= #

batch_size = 10
features = 25
x = torch.rand(batch_size, features)
print(x[0])
print(x[0].shape)  # x[0,:]

print(x[:, 0].shape)  # 10 - we have 10 examples in a batch

#say we want the third example in a batch and we want the first 5 features:
print(x[2, 0:5])

x[0,0] = 100

x = torch.arange(10)
print(x)  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
indices = [2,5,8]
print(x[indices])

x = torch.tensor(  [[1,2,3],[4,5,6],[7,8,9]]  )
rows = torch.tensor([1,0])  # second row and first row
cols = torch.tensor([2,0])  # third column and first column
print(x[rows, cols])        # tensor([6, 1])
assert(x[rows, cols].shape[0] == 2 )  #two elements: 6,1

# More advanced indexing...
x = torch.arange(10)
print(x[(x < 2) | (x > 8)] )  # tensor([0, 1, 9])
print(x[(x < 2) & (x > 8)] )  # tensor([])

print(x[x.remainder(2) == 0] )   #tensor([0, 2, 4, 6, 8])


# Useful operations
print(torch.where(x>5, x, x*2)) # tensor([ 0,  2,  4,  6,  8, 10,  6,  7,  8,  9])

print(torch.tensor([0,0,1,2,2,3,4]).unique())  # tensor([0, 1, 2, 3, 4])

print(x.ndimension())  #1 if the tensor is single dimention, 2 if it is two dimension, ect...

print(torch.rand([5,5,5]).ndimension())  #3

print(torch.rand([5,5,5]).numel())   # numel counts the number of elements in the tensor, 125 


