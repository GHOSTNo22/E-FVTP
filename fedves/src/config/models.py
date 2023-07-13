from torch import nn
import torch
ARGS = {
    "mnist": (1, 256, 10),
    "emnist": (1, 256, 62),
    "fmnist": (1, 256, 10),
    "cifar10": (3, 400, 10),
    "cifar100": (3, 400, 100),
}

# todo 更换模型
class LeNet5(nn.Module):
    def __init__(self, dataset) -> None:
        super(LeNet5, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ARGS[dataset][0], 6, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(ARGS[dataset][1], 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, ARGS[dataset][2]),
        )

    def forward(self, x):
        return self.net(x)

BATCH_SIZE=16

#4个时刻输入
class CNNGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=4,
                      out_channels=128,
                      kernel_size=2,
                      stride=1,
                      padding='valid'),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128,
                      out_channels=64,
                      kernel_size=2,
                      stride=1,
                      padding='valid'),
            nn.ReLU()
        )

        self.pool = nn.MaxPool1d(
            kernel_size=2,
        )

        self.output1 = nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,16)
        )

        self.GRU = nn.Sequential(
            nn.GRU(
                input_size=16,
                hidden_size=128,
                num_layers=2,
                bidirectional=True
            )
        )

        self.fc = nn.Linear(256,4)

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        # x = x.view(-1,256)
        x = x.view(x.shape[0],-1)
        x = self.output1(x)
        x = x.view(len(x),1,-1)
        x, _ = self.GRU(x)
        x = x.view(16,256)
        output = self.fc(x)
        return output


