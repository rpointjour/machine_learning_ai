# LeNet-5 Model: Convolutional Neural Network (CNN)
# 7 layers: 2 convolution layers, 2 pool layers, 3 fully connected layers
import torch
import torch.nn as nn   # nn module to inherit for Neural Network
import torch.nn.functional as F        # activation function

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Convolution Layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)    # Input = Outputs from previous conv & fc
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Pooling Layer : ?????????
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Display Neural Network information
net = LeNet()
print(net)

input = torch.rand(1, 1, 32, 32)   # 32x32 black & white image
print('\nImage batch shape: ')
print(input.shape)

output = net(input)
print('\nRaw output: ')
print(output)
print(output.shape)
