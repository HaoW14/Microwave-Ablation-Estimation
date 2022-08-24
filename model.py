import warnings

warnings.filterwarnings("ignore")

import torch.nn as nn
import torchvision
from tqdm import tqdm

import torch
from torch.utils.data import random_split, DataLoader

CUDA = torch.cuda.is_available()


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1

class Unet(nn.Module):  # resunet模型
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),  # 输入通道为1
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.decode4 = Decoder(512, 256 + 256, 256)
        self.decode3 = Decoder(256, 256 + 128, 256)
        self.decode2 = Decoder(256, 128 + 64, 128)
        self.decode1 = Decoder(128, 64 + 64, 64)

        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv_last = nn.Conv2d(32, n_class, 1)

        self.toseq = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.sequence = lstm(9)

    def forward(self, input, text):
        e1 = self.layer1(input)  # 64,128,128
        e2 = self.layer2(e1)  # 64,64,64
        e3 = self.layer3(e2)  # 128,32,32
        e4 = self.layer4(e3)  # 256,16,16
        e5 = self.layer5(e4)  # 512,8,8

        f = self.toseq(e5)
        f = torch.squeeze(f,2)
        f = torch.squeeze(f, 2)
        feature = self.fc(f)

        d4 = self.decode4(e5, e4)  # 256,16,16
        d3 = self.decode3(d4, e3)  # 256,32,32
        d2 = self.decode2(d3, e2)  # 128,64,64
        d1 = self.decode1(d2, e1)  # 64,128,128
        d0 = self.decode0(d1)  # 64,256,256
        out = self.conv_last(d0)  # 1,256,256


        inp = torch.cat([feature[:, None, :], text], 2)
        pred_time = self.sequence(inp)
        return pred_time, out

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

class lstm(nn.Module):
    def __init__(self, input_size, hidden_size=4, output_size=1,num_layers = 4):
        super(lstm, self).__init__()
        self.layer2 = nn.LSTM(input_size, 4, num_layers, batch_first= True)
        self.layer4 = nn.Linear(4, output_size) #加上1 + 6个临床特征做回归

    def forward(self, x):
        v2, _ = self.layer2(x)
        b, s, h = v2.size()
        v2 = v2.view(s * b, -1)
        v = self.layer4(v2)
        x = v.view(b, s, -1)
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                #tanh_gain = nn.init.calculate_gain('tanh')
                #nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)
                nn.init.kaiming_normal_(m.weight.data)

class lstm2(nn.Module):
    def __init__(self, output_size=1, num_layers=4):
        super(lstm2, self).__init__()
        self.layer2 = nn.LSTM(3, 2, num_layers, batch_first=True)
        self.layer4 = nn.Linear(2, output_size)

    def forward(self, x):
        v2, _ = self.layer2(x[:, :, :3])
        v = self.layer4(v2)
        return v