
# pip install torch
import torch
import numpy as np


# ======================================================================= #
#                      Initializing Tensor
# ======================================================================= #
device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32, device="cpu", requires_grad=True)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)  #cuda/cpu
print(my_tensor.shape)
print(my_tensor.requires_grad)

#other common initialization methods
x = torch.empty(size = (3,3))  #matrix with whatever was in the memory (usually bad)
print(x)
x = torch.zeros((3,3)) #matrix filled with zeros
print(x)
x = torch.rand((3,3))  #matrix with random variables
print(x)
x = torch.ones((3,3))  #matrix with all ones
print(x)
x=torch.eye(5,5)  # I, identity matrix
print(x)
x = torch.diag(torch.ones(3))  #creates matrix with ones on the diagonal (but could be anything on diag)
print(x)
x = torch.arange(start=0,end=5,step=1)  # tensor([0, 1, 2, 3, 4])
print(x)
x = torch.linspace(start=0.1, end=1, steps=10) #tensor([0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000, 1.0000])
print(x)


#init data from various distributions
x = torch.empty(size=(1,5)).normal_(mean=0, std=1)
print(x)
x = torch.empty(size=(1,5)).uniform_(0,1)
print(x)


print("****************** Converting Tensors ******************")

#how to initialize and convert tensors to another type (int, float, double)
tensor = torch.arange(4)
print(tensor.bool())   # True/False
print(tensor.short())  # int16
print(tensor.long())   # float64 ** important!
print(tensor.half())   # float16 - need 2000 series GPU
print(tensor.float())  # float32 -- really common
print(tensor.double()) # float64

np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array) # convert from np to torch
print(tensor)
np_array_back = tensor.numpy()  #could have rounding error, otherwise the same converted back to np
print(np_array_back)

print("****************** Math with Tensors ******************")
x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

#addition
z1 = torch.empty(3)
#torch.add(x, y, out=z1) #dumb syntax
z1 = torch.add(x,y)
print(z1)  #tensor([10, 10, 10])
z = x + y  #simple!
print(z)   #same as z1

#subtraction
z = x - y
print(z)

#multiplication - element wise
z = x * y
print(z)

#division
z = torch.true_divide(x, y)  #element wise division, clunky!
print(z)  # tensor([0.1111, 0.2500, 0.4286])

# inplace operations
t = torch.zeros(3)
t.add_(x)  #inplace addition, note the underscore, _
print(t)  

# Exponentiation
z = x.pow(2)
print(z)  # tensor([1, 4, 9])

# Compare
z = x > 2
print(z)  # [False, False,  True]


#matrix multiplication
x1 = torch.rand([2,5])
x2 = torch.rand([5,3])
x3 = torch.mm(x1,x2)
x3 = x1.mm(x2)   #same as line above
print(x3)   # tensor([[1.0649, 1.5045, 0.6939],
            #        [1.2010, 1.6988, 0.7816]])


matrix_exp = torch.rand(5,5)
mexp = matrix_exp.matrix_power(3)  #do the matrix multiplied by itself 3 times
print(mexp)

# dot product
x = torch.tensor([1,3,-5])
y = torch.tensor([4,-2,-1])
z = x.dot(y)    # [1,3,-5] . [4,-2,-1] = sum_{i=1 to n} (ai * bi) = a1b1 + a2b2 + ... + anbn
print(z)  # 3

# If vectors are identified with row matrices, the dot product can also be written as a matrix product
# a . b = ab^T   where ^T means transpose b

# Batch matrix multiply
batch = 2
n = 2
m = 3
p = 2

tensor1 = torch.rand((batch, n, m)) #2 5*m matrix
tensor2 = torch.rand((batch, m, p)) #2 m*p matrix
out_bmm = torch.bmm(tensor1, tensor2)  # (batch, n, p)

print(tensor1)
print("**")
print(tensor2)
print("**")
print(out_bmm)


print("*****")
print("*****")

# Example of Broadcasting
x1 = torch.tensor([[3,3,3],[4,4,4],[5,5,5]])
print(x1)
x2 = torch.tensor([1,2,3])  #expanded so it matches the rows of the first one
print(x2)
z = x1 - x2 
print("subtraction, broadcast:")
print(z)

z = x1 ** x2  #again, in element wise exponents, each row is copied so its the right shape, broadcasting again
print("exp, broadcast:")
print(z)

 #dim specifies which dimension we should sum over...
sum_0 = torch.sum(x1, dim=0) # columns
sum_1 = torch.sum(x1, dim=1) # rows
print(sum_0)  # tensor([12, 12, 12])  adds the columns
print(sum_1)  # tensor([ 9, 12, 15])  adds the rows

print("max, min, abs: ")
values, indices = torch.max(x1, dim=0) #column
print(values, indices)  # tensor([5, 5, 5]) tensor([2, 2, 2])
values, indices = torch.max(x1, dim=1) #row
print(values, indices)  # tensor([3, 4, 5]) tensor([0, 0, 0])
values, indices = torch.min(x1, dim=0) #column
print(values, indices)  # tensor([3, 3, 3]) tensor([0, 0, 0])
values, indices = torch.min(x1, dim=1) #row
print(values, indices)  # tensor([3, 4, 5]) tensor([0, 0, 0])
abs_x = torch.abs(x)
print(abs_x)            # tensor([1, 3, 5])
z = torch.argmax(x1, dim=0)  #just give back the index of the max (special case of above)
assert(z[0] == 2 and z[1] == 2 and z[2] == 2)
z = torch.argmin(x1, dim=0)  #same as argmax except does min instead of max
assert(z[0] == 0 and z[1] == 0 and z[2] == 0)
mean_x = torch.mean(x1.float(), dim=0)
print(mean_x)                #tensor([4., 4., 4.])
print(torch.eq(x1, x1))
torch.eq(x1, x1)        # element wise compare two vectors, a vector with true or false for each compare
compare_all = torch.all(x1.eq(x1)) #
assert(compare_all == True)

sorted_y, indices = torch.sort(y, dim=0, descending=False)
print(sorted_y, indices)

z = torch.clamp(x, min=0, max=2)  # all elements that are less than 0 in x will be set to zero (min argument)
                                  # all elements that are greater than 2 in x will be set to 2 (max argument)
print(z)  # tensor([1, 2, 0])


w = torch.tensor([1,0,0,0,1], dtype=torch.bool)
a = torch.any(w)
assert(a == True) #True, because some values in w are true...

