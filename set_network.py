import torch
from torch import nn


# Obtained from: https://github.com/manzilzaheer/DeepSets/blob/master/PointClouds/classifier.py#L58
class PermEqui1_mean(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm = x.mean(1, keepdim=True)
        x = self.Gamma(x-xm)
        return x


class E2ESetNetwork(nn.Module):
    def __init__(self, backbone, x_dim, d_dim, num_classes):
        super().__init__()

        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2]).eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.phi = self.phi = nn.Sequential(
            PermEqui1_mean(x_dim, d_dim),
            nn.ELU(inplace=True),
            PermEqui1_mean(d_dim, d_dim),
            nn.ELU(inplace=True),
            PermEqui1_mean(d_dim, d_dim),
            nn.ELU(inplace=True),
        )

        self.ro = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(d_dim, d_dim),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(d_dim, num_classes),
        )

    # Taken from SliverNet
    def nonadaptiveconcatpool2d(self, x, k):
    # concatenating average and max pool, with kernel and stride the same
        ap = torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=k)
        mp = torch.nn.functional.max_pool2d(x, kernel_size=k, stride=k)
        return torch.cat([mp, ap], 1)

    def forward(self, x):
        # TODO: SORT OUT THE DIMENSIONS SO IT WILL BE POSSIBLE TO DO IT WITH BxMxHxW
        batch_size, slices_num, channels, height, width = x.shape
        x = x.view(batch_size * slices_num, 3, height, width)

        if x.shape[0] > 100:  # Cuda & ResNet are having trouble with long vectors, so split
            split = torch.split(x, 100)
            temp_features = []
            for chunk in split:
                temp_features.append(self.backbone(chunk))
            features = torch.cat(temp_features)
        else:
            features = self.backbone(x)  # B x M x h x w - B=batch size, M=#slices_per_volume, h=height, w=width

        kernel_size = (features.shape[-2], features.shape[-1])
        features = self.nonadaptiveconcatpool2d(features, kernel_size).view(batch_size, slices_num, -1)

        phi_output = self.phi(features)
        sum_output = phi_output.mean(1)
        ro_output = self.ro(sum_output)
        return ro_output
