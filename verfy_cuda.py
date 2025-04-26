import torch
import torch.nn as nn
from torch.nn import DataParallel


class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.fc1 = nn.Linear(in_features=512, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=128)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = TestModule().to(device)
    # model = DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    input = torch.randn(size=(10000, 512), device=device)
    power = torch.randn(size=(10000, 128), device=device)
    model.to(device)
    optmizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for i in range(10):
        y = model(input)
        loss = torch.mean(y * power)
        optmizer.zero_grad()
        loss.backward()
        optmizer.step()
        print(i, loss.item())
