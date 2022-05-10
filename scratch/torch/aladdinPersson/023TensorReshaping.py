import torch
import numpy as np


# ======================================================================= #
#                      Tensor Reshaping
# ======================================================================= #

x = torch.arange(9)  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
print(x)
print(x.view(-1))    # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
print(x.view([3,3])) # tensor([[0, 1, 2],
                     #         [3, 4, 5],
                     #         [6, 7, 8]])
print(x.reshape([3,3])) #same as above

#view acts on contigous tensors- something stored contigiously in memory
#reshape doesn't really matter, but if it is not it will make a copy... so can be a performance loss

y = x.view([3,3]).t() # take the transpose of x
print(y)              # tensor([[0, 3, 6],
                      #         [1, 4, 7],
                      #         [2, 5, 8]])

#y.view(9)  #error!!! its not a continous block of memory!!!
y_cont = y.contiguous()
print(y_cont.view(9))  # works!  tensor([0, 3, 6, 1, 4, 7, 2, 5, 8])

print(y.reshape(9))    #also works, but slower tensor([0, 3, 6, 1, 4, 7, 2, 5, 8])

#cat - concatinate tensors together
x1 = torch.tensor([[1,2,3,4,5],[6,7,8,9,10]])
x2 = torch.tensor([[2,4,6,8,10],[12,14,16,18,20]])


#create this with cat:
#tensor([[ 1,  2,  3,  4,  5],
#        [ 6,  7,  8,  9, 10],
#        [ 2,  4,  6,  8, 10],
#        [12, 14, 16, 18, 20]])
print(torch.cat((x1,x2), dim=0) ) 

#cat the other way...
#tensor([[ 1,  2,  3,  4,  5,  2,  4,  6,  8, 10],
#        [ 6,  7,  8,  9, 10, 12, 14, 16, 18, 20]])
print(torch.cat((x1,x2), dim=1) ) 

#unroll a tensor
z = x1.view(-1)
assert(z.shape[0] == 10)  #10  - unrolls all elements into a single vector

batch = 3
z = x.view(batch, -1)  #unroll by a batch size!
print(z)
#tensor([[0, 1, 2],
#        [3, 4, 5],
#        [6, 7, 8]])  #notice how last elements are truncated!

#What about transposing a matrix?  thats just a special case of the permute function... can use .t() as above, or:
#transpose x1
#tensor([[ 1,  6],
#        [ 2,  7],
#        [ 3,  8],
#        [ 4,  9],
#        [ 5, 10]])
print(x1.permute([1, 0]))

w = torch.randn(2, 3, 5)
w.size() # torch.Size([2, 3, 5])
print(w)
wp = w.permute(2,0,1).size()  #note how permute takes the dimentions that we want as a list of arguments, 2->0,0->1,1->2
print(wp)

x = torch.arange(10)
print(x)                 # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(x.unsqueeze(0))    # tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])   note the additional brackets
print(x.unsqueeze(1))    # tensor([ [0],
                         #          [1],
                         #          [2],
                         #          [3],
                         #          [4],
                         #          [5],
                         #          [6],
                         #          [7],
                         #          [8],
                         #          [9]])

x = torch.arange(10).unsqueeze(0).unsqueeze(1)  #1x1x10
print(x)  # tensor([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]])
print(x.shape)
print(x.squeeze(0).squeeze(0))  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  squeeze(1) also works... eleminates extra brackets

