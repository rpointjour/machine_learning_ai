# Use PyTorch!
# Use Matplot Library

import torch
import matplotlib.pyplot as plt
import numpy as np

# Graph random tensor
torch.manual_seed(1733)
rn1 = torch.rand(1,5)
torch.manual_seed(1733)
rn2 = torch.rand(1,5) * 2

xtorch = np.array(rn1)
ytorch = np.array(rn2)

print(rn1)
print(rn2)

plt.scatter(xtorch,ytorch)
plt.show()
