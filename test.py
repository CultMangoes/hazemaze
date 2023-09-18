import torch
import torch.nn as nn


# Define Model A and B
class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.linear = nn.Linear(5, 1)

    def forward(self, x):
        return self.linear(x)


# Instantiate models A and B
model_A = ModelA()
model_B = ModelB()

# Set requires_grad to False for parameters of model B
for param in model_B.parameters():
    param.requires_grad = False

# Define some input
x = torch.randn(10)

# Forward pass
output = model_B(model_A(x))

# Define loss functions
loss1 = torch.mean(output ** 2)
loss2 = torch.mean(output ** 3)

# Backward pass for loss1 (only model A will accumulate gradients)
loss1.backward()

# Set requires_grad back to True for parameters of model B
for param in model_B.parameters():
    param.requires_grad = True

# Backward pass for loss2 (both model A and model B will accumulate gradients)
loss2.backward()

# Check gradients
print(model_A.linear.weight.grad)
print(model_B.linear.weight.grad)
