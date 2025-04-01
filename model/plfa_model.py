import torch
import torch.nn as nn


class PhaseLayer(nn.Module):
    def __init__(self, in_c):
        super(PhaseLayer, self).__init__()
        if (in_c == 2):
            self.conv = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 1), stride=(1, 1))
        else:
            self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(2, 1), stride=(2, 1))
        self.bn = nn.BatchNorm2d(num_features=4)
        self.relu = nn.ReLU()
        self.max_norm = 0.5

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        nn.utils.clip_grad_norm_(self.conv.parameters(), self.max_norm, 2)
        return x


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.tanh = nn.Tanh()


    def forward(self, x):
        source = x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = self.tanh(x)
        x = x * source
        x = x + source
        return x


class SpaceFrequenceLayer(nn.Module):
    def __init__(self):
        super(SpaceFrequenceLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(5, 5), stride=(1, 1))
        self.space_fre = nn.Sequential(
            nn.BatchNorm2d(num_features=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(p=0.5)
        )
        self.max_norm = 0.5

    def forward(self, x):
        x = self.conv(x)
        x = self.space_fre(x)
        return x


class ClassificationLayer(nn.Module):
    def __init__(self, num_classes, flatten_dim):
        super(ClassificationLayer, self).__init__()
        self.max_norm = 0.5
        self.num_classes = num_classes
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(flatten_dim, 512)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        nn.utils.clip_grad_norm_(self.linear1.parameters(), self.max_norm, 2)
        x = self.relu1(x)
        x = self.linear2(x)
        nn.utils.clip_grad_norm_(self.linear2.parameters(), self.max_norm, 2)
        x = self.relu2(x)
        x = self.linear3(x)
        nn.utils.clip_grad_norm_(self.linear3.parameters(), self.max_norm, 2)
        return x


class PLFA(nn.Module):
    def __init__(self, num_classes, in_c, img_size):
        super(PLFA, self).__init__()
        self.num_classes = num_classes
        self.phase_layer = PhaseLayer(in_c)
        self.spatial_attn = SpatialAttention()
        self.space_fre_layer = SpaceFrequenceLayer()
        if img_size[0] == 8:
            self.flatten_dim = 1008
        if img_size[0] == 30:
            self.flatten_dim = 6552
        self.classification_layer = ClassificationLayer(self.num_classes, self.flatten_dim)

    def forward(self, x):
        x = self.phase_layer(x)
        x = self.spatial_attn(x)
        x = self.space_fre_layer(x)
        x = self.classification_layer(x)
        return x


def make_model(args):
    if args.dataset_name == 'BETA' or args.dataset_name == 'Benchmark':
        model = PLFA(num_classes=args.num_classes, in_c=args.in_c, img_size=(30, 256))
    elif args.dataset_name == 'JFPM':
        model = PLFA(num_classes=args.num_classes, in_c=args.in_c, img_size=(8, 256))
    else:
        return None
    return model

