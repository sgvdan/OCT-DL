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
    def __init__(self, backbone, x_dim, d_dim):
        super().__init__()

        self.backbone = backbone
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
            nn.Linear(d_dim, 40),
        )

    def forward(self, x):
        features = self.backbone(x)  # B x M x h x w - B=batch size, M=#slices_per_volume, h=height, w=width
        phi_output = self.phi(features)
        sum_output = phi_output.mean(1)
        ro_output = self.ro(sum_output)
        return ro_output
