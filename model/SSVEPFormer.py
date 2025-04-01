import torch
import torch.nn as nn

class SSVEPEncoder(nn.Module):
    def __init__(self, encoder_layer):
        super(SSVEPEncoder, self).__init__()
        self.layers = nn.Sequential(*[
            encoder_layer
            for _ in range(2)
        ])
    def forward(self, x):
        x = self.layers(x)
        return x


class SSVEPEncoder_layer(nn.Module):
    def __init__(self, img_size):
        super(SSVEPEncoder_layer, self).__init__()
        self.cnn = nn.Conv1d(img_size[0] * 2, img_size[0] * 2, kernel_size=31, stride=1, padding='same')
        self.layerNorm = nn.LayerNorm(img_size[1] * 2)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.5)

        self.linear = nn.Linear(img_size[1] * 2, img_size[1] * 2)
    def forward(self, x):
        x = x + self.dropout(self.gelu(self.layerNorm(self.cnn(self.layerNorm(x)))))
        x = x + self.dropout(self.gelu(self.linear(self.layerNorm(x))))
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.5):
        super(Mlp, self).__init__()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=drop)
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.layer_norm = nn.LayerNorm(hidden_features)
        self.act = act_layer()
        self.linear2 = nn.Linear(hidden_features, out_features)
    def forward(self, x):
        x = self.dropout(self.flatten(x))
        x = self.linear2(self.dropout(self.act(self.layer_norm(self.linear1(x)))))
        return x


class SSVEPFormer(nn.Module):
    def __init__(self, num_classes, img_size):
        super(SSVEPFormer, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.channel_combination = nn.Sequential(
            nn.Conv1d(img_size[0], img_size[0] * 2, 1),
            nn.LayerNorm(img_size[1] * 2),
            nn.GELU(),
            nn.Dropout(p=0.5)
        )
        self.encoder_layer = SSVEPEncoder_layer(img_size)
        self.encoder = SSVEPEncoder(self.encoder_layer)
        self.mlp = Mlp(in_features=img_size[1] * 2 * img_size[0] * 2, hidden_features=num_classes * 6, out_features=num_classes)

    def forward(self, x):
        x = self.channel_combination(torch.cat([x[:, 0, :, :], x[:, 1, :, :]], dim=2))
        x = self.encoder(x)
        x = self.mlp(x)
        return x


def make_model(args):
    if args.dataset_name == 'BETA' or args.dataset_name == 'Benchmark':
        model = SSVEPFormer(num_classes=args.num_classes, img_size=(30, 256))
    elif args.dataset_name == 'JFPM':
        model = SSVEPFormer(num_classes=args.num_classes, img_size=(8, 256))
    else:
        return None
    return model