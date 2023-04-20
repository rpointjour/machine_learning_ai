# Introduction to PyTorch Machine Learning framework
# Machine Learning concepts: Tensors
# Tensors = 'multi-dimensional arrays' / matrices

import torch
# 3 ways to initialize tensors
# 1) Default data type filled with zeros
first_tensor = torch.zeros(5, 3)     # Create 5x3 matrix filled with zeros!
print(first_tensor)         # Display tensor
print(first_tensor.dtype)       # Show tensor datatype : float32 by default!

# 2) Desired data type with override: Integer Tensor
itn = torch.ones((2,2), dtype = torch.int16)      # 2x2 matrix of int16 values
print(itn)
# 3 As a number generator
torch.manual_seed(1735)
r1 = torch.rand(3,4)
print('Random tensor 1: ')
print(r1)
r2 = torch.rand(3,4)
print('Random tensor 2: ')
print(r2)
torch.manual_seed(1735)
r3 = torch.rand(3,4)
print('Random 3 should equal random 1: ')
print(r3)
# Next --> Tensor Arithmetic: ONLY Tensors of similar shapes 'size' can be operated on
# Operations only apply to each element inside the tensors!!
# Addition
xten = torch.ones(3,3)
print('Before addition operation..')
print(xten)
yten = torch.ones(3,3) + 4      # Add each element inside tensor by 4
print('After adding 4 to each element!')
print(yten)
zten = xten + yten  # Allowed, each tensor is added element-wise!
print('After adding both tensors...')
print(zten)
# Uncomment to see error!
# aten = torch.ones(2,4)
# bten = torch.ones(3,3)
# cten = aten + bten
# print(cten)
# Subtraction
gten = torch.ones(2,2)
hten = torch.ones(2,2) + 8
jten = hten - gten
print('Subtraction should result in a tensor of 8s')
print(jten)
# Multiplication
mten = torch.ones(4,3) * 2
pten = torch.ones(4,3) * 10
nten = mten * pten
print('Product should result in a tensor of 20s')
print(nten)
# Division
qten = torch.ones(2,4) + 7
wten = torch.ones(2,4) * 24
uten = wten / qten
print('Quotient should result in a tensor of 3s')
print(uten)
# Mod
e = torch.ones(1,2) + 8
f = torch.ones(1,2) + 4
d = e % f         # 9 mod 5
print('Mod should be 4s')
print(d)
# With rand
torch.manual_seed(1639)
star = torch.rand(3,3)
gen = torch.rand(3,3)
trooper = star + gen
print('Random results')
print(trooper)
