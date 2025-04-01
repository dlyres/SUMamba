import torch
import torch.nn as nn

class CCNN(nn.Module):
    def __init__(self, num_classes, in_c, img_size):
        super(CCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 2 * img_size[0], kernel_size=(img_size[0], 1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(2 * img_size[0])
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2d(2 * img_size[0], 2 * img_size[0], kernel_size=(1, 10), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(2 * img_size[0])
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2 * img_size[0] * (img_size[1] - 9), num_classes)

    def forward(self, x):
        x = self.dropout(self.bn1(self.conv1(x)))
        x = self.dropout(self.bn2(self.conv2(x)))
        x = self.linear(self.flatten(x))
        return x


def make_model(args):
    if args.dataset_name == 'BETA' or args.dataset_name == 'Benchmark':
        model = CCNN(num_classes=args.num_classes, in_c=args.in_c, img_size=(30, 256))
    elif args.dataset_name == 'JFPM':
        model = CCNN(num_classes=args.num_classes, in_c=args.in_c, img_size=(8, 256))
    else:
        return None
    return model