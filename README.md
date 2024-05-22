# pyflashlight
Recreating PyTorch from scratch (C/C++, CUDA and Python, with GPU support and automatic differentiation!)
![2671157D-5A4D-4D09-A462-7D45B024D6B0](https://github.com/Eeman1113/PyFlashlight/assets/54275491/d51e1ad7-0e73-466a-a461-577bf2614e91)

Project details explanations can also be found on [medium](https://medium.com/@eeman.majumder/i-made-pytorch-02238e268b63).

# 1 - About
**pyflashlight** is a deep learning framework constructed using C/C++, CUDA and Python. This is a personal project with educational purpose only! `pyflashlight` means **NOT** PyTorch, and we have **NO** claims to rivaling the already established PyTorch. The main objective of **pyflashlight** was to give a brief understanding of how a deep learning framework works internally. It implements the Tensor object, GPU support and an automatic differentiation system. 

# 2 - Installation
Install this package from PyPi (you can test on Colab!)

```css
$ pip install pyflashlight
```

or from cloning this repository
```css
$ sudo apt install nvidia-cuda-toolkit
$ git clone https://github.com/lucasdelimanogueira/pyflashlight.git
$ cd build
$ make
$ cd ..
```

# 3 - Get started
### 3.1 - Tensor operations
```python
import pyflashlight

x1 = pyflashlight.Tensor([[1, 2], 
                  [3, 4]], requires_grad=True).to("cuda")

x2 = pyflashlight.Tensor([[4, 3], 
                  [2, 1]], requires_grad=True).to("cuda)

x3 = x1 @ x2
result = x3.sum()
result.backward

print(x1.grad)
```

### 3.2 - Create a model

```python
import pyflashlight
import pyflashlight.nn as nn
import pyflashlight.optim as optim

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

### 3.3 - Example training
```python
import pyflashlight
from pyflashlight.utils.data.dataloader import Dataloader
from pyflashlight.pyflashlightvision import transforms
import pyflashlight
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

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss[0]:.4f}')
    loss_list.append(loss[0])

```


# 4 - Progress

| Development                  | Status      | Feature                                                                |
| ---------------------------- | ----------- | ---------------------------------------------------------------------- |
| Operations                   | in progress | <ul><li>[X] GPU Support</li><li>[X] Autograd</li><li>[X] Broadcasting</li></ul>                 |
| Loss                         | in progress | <ul><li>[x] MSE</li><li>[X] Cross Entropy</li></ul>    |
| Data                         | in progress    | <ul><li>[X] Dataset</li><li>[X] Batch</li><li>[X] Iterator</li></ul>   |
