import torch
import torch.nn as nn


class Recalibration(nn.Module):
    def __init__(self, img_size):
        super(Recalibration, self).__init__()
        self.fre_avg = nn.AvgPool2d(kernel_size=(img_size[0], 1), stride=(1, 1))
        self.fre_linear1 = nn.Linear(img_size[1], img_size[1] // 4)
        self.fre_linear2 = nn.Linear(img_size[1] // 4, img_size[1])

        self.channel_avg = nn.AvgPool2d(kernel_size=(img_size[1], 1), stride=(1, 1))
        self.channel_linear1 = nn.Linear(img_size[0], img_size[0] // 4)
        self.channel_linear2 = nn.Linear(img_size[0] // 4, img_size[0])

    def forward(self, x):
        x_fre = self.fre_linear2(self.fre_linear1(self.fre_avg(x)))
        x_fre = x * x_fre
        x_channel = self.channel_linear2(self.channel_linear1(self.channel_avg(x.transpose(2, 3)))).transpose(2, 3)
        x_channel = x * x_channel
        return x_fre + x_channel

class Add(nn.Module):
    def __init__(self, img_size):
        super(Add, self).__init__()
        self.channel_conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(img_size[0], 1), stride=(1, 1))
        self.fre_conv1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 10), stride=(1, 1), padding='same')
        self.channel_conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(img_size[0], 1), stride=(1, 1))
        self.fre_conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 5), stride=(1, 1), padding='same')
    def forward(self, x):
        x = torch.cat([self.fre_conv1(self.channel_conv1(x)), self.fre_conv2(self.channel_conv2(x))], dim=1)
        return x

class MS1D_CNN(nn.Module):
    def __init__(self, num_classes, img_size):
        super(MS1D_CNN, self).__init__()
        self.recalibration = Recalibration(img_size)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=(1, 1), stride=(1, 1))
        self.add = Add(img_size)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(8192, num_classes)

    def forward(self, x):
        x = self.recalibration(x)
        x = self.conv1(x)
        x = self.add(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


def make_model(args):
    if args.dataset_name == 'BETA' or args.dataset_name == 'Benchmark':
        model = MS1D_CNN(num_classes=args.num_classes, img_size=(30, 256))
    elif args.dataset_name == 'JFPM':
        model = MS1D_CNN(num_classes=args.num_classes, img_size=(8, 256))
    else:
        return None
    return model