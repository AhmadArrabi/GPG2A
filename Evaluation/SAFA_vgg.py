import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet34(pretrained=True)
        layers = list(net.children())[:3]
        layers_end = list(net.children())[4:-3]
        self.layers = nn.Sequential(*layers, *layers_end)

    def forward(self, x):
        return self.layers(x)

class SA(nn.Module):
    def __init__(self, in_dim, num=8):
        super().__init__()
        hid_dim = in_dim // 2
        self.w1, self.b1 = self.init_weights_(in_dim, hid_dim, num)
        self.w2, self.b2 = self.init_weights_(hid_dim, in_dim, num)

    def init_weights_(self, din, dout, dnum):
        weight = torch.empty(din, dout, dnum)
        nn.init.normal_(weight, mean=0.0, std=0.005)
        bias = torch.empty(1, dout, dnum)
        nn.init.constant_(bias, val=0.1)
        weight = torch.nn.Parameter(weight)
        bias = torch.nn.Parameter(bias)
        return weight, bias

    def forward(self, x):
        mask, _ = x.max(1)
        mask = torch.einsum('bi, ijd -> bjd', mask, self.w1) + self.b1
        mask = torch.einsum('bjd, jid -> bid', mask, self.w2) + self.b2
        return mask

class SAFA_vgg(nn.Module):
    def __init__(self, safa_heads = 1): # always no polar transformation for VIGOR
        super().__init__()
        self.backbone_grd = models.vgg16(pretrained=True)
        self.backbone_sat = models.vgg16(pretrained=True)

        feats_list = list(self.backbone_grd.features)
        feats_list = feats_list[:-1]
        new_feats_list = []
        for i in range(len(feats_list)):
            new_feats_list.append(feats_list[i])
            if isinstance(feats_list[i], nn.Conv2d) and i > 14:
                new_feats_list.append(nn.Dropout(p=0.2, inplace=True))
        self.backbone_grd.features = nn.Sequential(*new_feats_list)

        modules=list(self.backbone_grd.children())
        modules = modules[:len(modules) - 2]
        self.backbone_grd = nn.Sequential(*modules)

        feats_list = list(self.backbone_sat.features)
        feats_list = feats_list[:-1]
        new_feats_list = []
        for i in range(len(feats_list)):
            new_feats_list.append(feats_list[i])
            if isinstance(feats_list[i], nn.Conv2d) and i > 14:
                new_feats_list.append(nn.Dropout(p=0.2, inplace=True))
        self.backbone_sat.features = nn.Sequential(*new_feats_list)

        modules=list(self.backbone_sat.children())
        modules = modules[:len(modules) - 2]
        self.backbone_sat = nn.Sequential(*modules)

        self.spatial_aware_grd = SA(in_dim=800, num=safa_heads)
        self.spatial_aware_sat = SA(in_dim=400, num=safa_heads)

        self.tanh = nn.Tanh()

    def forward(self, sat, grd, is_cf):
        sat_x = self.backbone_sat(sat)
        grd_x = self.backbone_grd(grd)

        sat_x = sat_x.view(sat_x.shape[0], sat_x.shape[1],-1)
        grd_x = grd_x.view(grd_x.shape[0], grd_x.shape[1],-1)
        sat_sa = self.spatial_aware_sat(sat_x)
        grd_sa = self.spatial_aware_grd(grd_x)
        sat_sa = self.tanh(sat_sa)
        grd_sa = self.tanh(grd_sa)
        if is_cf:
            fake_sat_sa = torch.zeros_like(sat_sa).uniform_(-1, 1)
            fake_grd_sa = torch.zeros_like(grd_sa).uniform_(-1, 1)

            sat_global = torch.matmul(sat_x, sat_sa).view(sat_x.shape[0], -1)
            grd_global = torch.matmul(grd_x, grd_sa).view(grd_x.shape[0], -1)

            sat_global = F.normalize(sat_global, p=2, dim=1)
            grd_global = F.normalize(grd_global, p=2, dim=1)

            fake_sat_global = torch.matmul(sat_x, fake_sat_sa).view(sat_x.shape[0], -1)
            fake_grd_global = torch.matmul(grd_x, fake_grd_sa).view(grd_x.shape[0], -1)

            fake_sat_global = F.normalize(fake_sat_global, p=2, dim=1)
            fake_grd_global = F.normalize(fake_grd_global, p=2, dim=1)

            return sat_global, grd_global, fake_sat_global, fake_grd_global
        else:
            sat_global = torch.matmul(sat_x, sat_sa).view(sat_x.shape[0], -1)
            grd_global = torch.matmul(grd_x, grd_sa).view(grd_x.shape[0], -1)

            sat_global = F.normalize(sat_global, p=2, dim=1)
            grd_global = F.normalize(grd_global, p=2, dim=1)

            return sat_global, grd_global



class SAFA_vgg_sat(nn.Module):
    def __init__(self, safa_heads = 1): # always no polar transformation for VIGOR
        super().__init__()
        self.num_features = safa_heads * 512
        self.backbone_sat = models.vgg16(pretrained=True)

        feats_list = list(self.backbone_sat.features)
        feats_list = feats_list[:-1]
        new_feats_list = []
        for i in range(len(feats_list)):
            new_feats_list.append(feats_list[i])
            if isinstance(feats_list[i], nn.Conv2d) and i > 14:
                new_feats_list.append(nn.Dropout(p=0.2, inplace=True))
        self.backbone_sat.features = nn.Sequential(*new_feats_list)

        modules=list(self.backbone_sat.children())
        modules = modules[:len(modules) - 2]
        self.backbone_sat = nn.Sequential(*modules)

        self.spatial_aware_sat = SA(in_dim=400, num=safa_heads)

        self.tanh = nn.Tanh()

    def forward(self, sat):
        sat_x = self.backbone_sat(sat)

        sat_x = sat_x.view(sat_x.shape[0], sat_x.shape[1],-1)

        sat_sa = self.spatial_aware_sat(sat_x)

        sat_sa = self.tanh(sat_sa)
        sat_global = torch.matmul(sat_x, sat_sa).view(sat_x.shape[0], -1)

        sat_global = F.normalize(sat_global, p=2, dim=1)

        return sat_global