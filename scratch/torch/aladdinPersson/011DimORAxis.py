import torch

# one thing thats quite confusing when working with torch is the idea of DIM or axis
# This article does a good job of explaining it:
# https://towardsdatascience.com/understanding-dimensions-in-pytorch-6edf9972d3be

#simple example
x = torch.tensor([1, 2, 3])
s = torch.sum(x)
assert(6 == s)

## Official 
## torch.sum(input, dim, keepdim=False, dtype=None) → Tensor
## Returns the sum of each row of the input tensor in the given dimension dim.
## What the heck does this mean?
## example:

x = torch.tensor([
     [1, 2, 3],
     [4, 5, 6]
   ])

print(x.shape)  # torch.Size([2, 3])


# The way to understand the 'dim' of torch sum is that it collapses the specified axis. 
# So when it collapses the axis/dim 0 (the row), it becomes just one row (it sums column-wise).
# When it collapses the axis/dim 1 (the column), it becomes just one column (it sums row-wise).

# e.g.
dim0 = torch.sum(x, dim=0)  # collapse rows
print(dim0) # tensor([5, 7, 9])

dim1 = torch.sum(x, dim=1)  # collapse columns
print(dim1) # tensor([ 6, 15])

# you can also pass two dims to the function to collapse BOTH rows and columns
dim0_1 = torch.sum(x, dim=[0,1])
print(dim0_1)  # tensor(21)

## 3d - 3d gets more confusing, the z axis takes over as dim=0.
y = torch.tensor([
     [
       [1, 2, 3],
       [4, 5, 6]
     ],
     [
       [7, 8, 9],
       [10, 11, 12]
     ],
     [
       [13, 14, 15],
       [16, 17, 18]
     ]
   ])
print(y.shape)  # torch.Size([3, 2, 3])

# sum and collapse along the z axis (matrix wise)
y0 = torch.sum(y, dim=0)
print(y0)    # tensor([[21, 24, 27],
             #         [30, 33, 36]])

# sum and collapse row wise
y1 = torch.sum(y, dim=1)
print(y1)  # tensor([[ 5,  7,  9],
            #       [17, 19, 21],
            #       [29, 31, 33]])

# sum and collapse column wise
y2 = torch.sum(y, dim=2)
print(y2)  # tensor([[ 6, 15],
            #        [24, 33],
            #        [42, 51]])

# point is, the last dimention is always column-wise
# second to last dimention is always row-wise ect.