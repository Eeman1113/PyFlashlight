
# PyFlashlight

![PyFlashlight Logo](https://github.com/Eeman1113/PyFlashlight/assets/54275491/d51e1ad7-0e73-466a-a461-577bf2614e91)

## About

**PyFlashlight** is a deep learning framework constructed using C/C++, CUDA, and Python. This personal project serves educational purposes only. It is not intended to rival the established PyTorch framework. The main objective of PyFlashlight is to provide insight into how a deep learning framework operates internally. It implements essential components such as the Tensor object, GPU support, and an automatic differentiation system.

## Installation

Clone this repository and build PyFlashlight:

```bash
$ sudo apt install nvidia-cuda-toolkit
$ git clone clone https://github.com/Eeman1113/PyFlashlight.git
$ cd build
$ make
$ cd ..
```

## Getting Started

### 1. Tensor Operations

```python
import pyflashlight

x1 = pyflashlight.Tensor([[1, 2], 
                  [3, 4]], requires_grad=True).to("cuda")

x2 = pyflashlight.Tensor([[4, 3], 
                  [2, 1]], requires_grad=True).to("cuda")

x3 = x1 @ x2
result = x3.sum()
result.backward

print(x1.grad)
```

### 2. Creating a Model

```python
import pyflashlight
import pyflashlight.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        
        return out
```

### 3. Example Training

```python
import pyflashlight
from pyflashlight.utils.data.dataloader import Dataloader
from pyflashlight.pyflashlightvision import transforms
import pyflashlight.nn as nn
import pyflashlight.optim as optim
import random

random.seed(1)

BATCH_SIZE = 32
device = "cpu"
epochs = 10

transform = transforms.Sequential(
    [
        transforms.ToTensor(),
        transforms.Reshape([-1, 784, 1])
    ]
)

target_transform = transforms.Sequential(
    [
        transforms.ToTensor()
    ]
)

train_data, test_data = pyflashlight.pyflashlightvision.datasets.MNIST.splits(transform=transform, target_transform=target_transform)
train_loader = Dataloader(train_data, batch_size = BATCH_SIZE)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 30)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(30, 10)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid1(out)
        out = self.fc2(out)
        out = self.sigmoid2(out)
        
        return out

model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_list = []

for epoch in range(epochs):    
    for idx, batch in enumerate(train_loader):

        inputs, target = batch

        inputs = inputs.to(device)
        target = target.to(device)

        outputs = model(inputs)
        
        loss = criterion(outputs, target)
        
        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    loss_list.append(loss.item())
```

# Progress

| Development                  | Status      | Feature                                                                |
| ---------------------------- | ----------- | ---------------------------------------------------------------------- |
| Operations                   | in progress | <ul><li>[X] GPU Support</li><li>[X] Autograd</li><li>[X] Broadcasting</li></ul>                 |
| Loss                         | in progress | <ul><li>[x] MSE</li><li>[X] Cross Entropy</li></ul>    |
| Data                         | in progress    | <ul><li>[X] Dataset</li><li>[X] Batch</li><li>[X] Iterator</li></ul>   |

