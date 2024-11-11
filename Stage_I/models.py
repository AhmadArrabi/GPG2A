import torch
import torchvision.models as models
import torch.nn as nn
            
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        x = self.up(x)
        x = torch.topk(x, self.out_channels, dim=1).values
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class UpsampleConv(nn.Module): #conv and upsample
    def __init__(self, in_channels, out_channels):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        
        self.same_channel_dim = in_channels==out_channels
        
        if self.same_channel_dim:
            #no need to align input for residual connection
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
                nn.InstanceNorm2d(out_channels)
            )
        else:
            #need to make input channel dimension same as after conv
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.conv_align = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.relu = nn.ReLU()            
        
    def forward(self, x): 
        if self.same_channel_dim:
            return x + self.conv2(self.conv1(x))
        else:
            return self.conv_align(x) + self.conv2(self.conv1(x))


class MLSP(nn.Module):
    def __init__(self, 
                 backbone_channels=256, 
                 classes=8):
        
        super().__init__()
        
        self.residual_block = nn.Sequential(
            ResBlock(backbone_channels, backbone_channels//2), 
            ResBlock(backbone_channels//2, backbone_channels//2)) #256,32,32
        
        self.conv_upsample1 = UpsampleConv(backbone_channels/2, backbone_channels/4) #64,64,64
        self.upsample1 = Upsample(backbone_channels/2, backbone_channels/4) #64,64,64

        self.conv_upsample2 = UpsampleConv(backbone_channels/2, backbone_channels/8) #32,128,128
        self.upsample2 = Upsample(backbone_channels/4, backbone_channels/8) #32,128,128

        self.conv_out = DoubleConv(backbone_channels//4, backbone_channels//8)
        self.seg = nn.Conv2d(backbone_channels//8, classes, kernel_size=1, stride=1, padding=0)
                   
    def forward(self, x):
        x = nn.functional.relu(self.residual_block(x)) #256,32,32
        
        x_conv_up1 = self.conv_upsample1(x) #64,64,64
        x_up1 = self.upsample1(x) #64,64,64
        x = torch.cat([x_conv_up1, x_up1], dim=1) #128,64,64
        
        x_conv_up2 = self.conv_upsample2(x) #32,128,128
        x_up2 = self.upsample2(x_up1) #32,128,128
        x = torch.cat([x_conv_up2, x_up2], dim=1)#64,128,128
        
        x = self.conv_out(x)
        x = self.seg(x)
        
        return x


class ConvNextBase(nn.Module):
    def __init__(self, out_channel=None):
        super().__init__()
        net = models.convnext_base(weights='IMAGENET1K_V1')
        layers = list(net.children())[:-2]
        layers = list(layers[0].children())[:-2]
        self.net = torch.nn.Sequential(*layers, nn.Conv2d(512, out_channel, 1))
    
    def forward(self, x):
        return self.net(x) # (B, 384, H/16, W/16) -> (B, out_channel, H/16, /w/16)
    

class BEV_projection(nn.Module):
    def __init__(self, 
                 backbone_channels=256, 
                 depth_channels=64, 
                 bev_projection_size=32, 
                 device='cpu'):
        
        super().__init__()
        
        self.device=device

        self.backbone_channels = backbone_channels
        self.depth_channels = depth_channels
        self.bev_projection_size = bev_projection_size
        
        self.Backbone = ConvNextBase(out_channel=backbone_channels)
        self.depth_conv = nn.Conv2d(backbone_channels, backbone_channels*depth_channels, 1)
        self.init_cartesian_grid()

    def forward(self, x): #256x512 -> 16x32 -> dxw = 64x32
        f_g = self.Backbone(x) #(B, C, H, W)
        B, C, H, W = f_g.shape

        # create learnable depth dimension 
        depth_weights = self.depth_conv(f_g) #(B, DxC, H, W)
        depth_weights = depth_weights.unsqueeze(1) #(B, 1, DxC, H, W)
        depth_weights = depth_weights.view(B, self.depth_channels, self.backbone_channels, H, W) #(B, D, C, H, W)
        depth_weights = torch.permute(depth_weights, (0,2,1,3,4)) #(B, C, D, H, W)
        depth_weights_normalized = nn.functional.softmax(depth_weights, dim=3) #(B, C, D, H, W)
        f_g = f_g.unsqueeze(2) #(B, C, 1, H, W)
        f_polar = f_g*depth_weights_normalized #(B, C, D, H, W)
        f_polar = f_polar.sum(dim=3) #(B, C, D, W)
        f_bev = self.cartesian_proj(f_polar)
        return f_bev
    
    def init_cartesian_grid(self):
        x = torch.linspace(-1, 1, self.bev_projection_size, device=self.device)
        y = torch.linspace(-1, 1, self.bev_projection_size, device=self.device)

        meshy, meshx = torch.meshgrid((x, y), indexing='ij')

        mesh_r = (meshx**2 + meshy**2).sqrt()
        mesh_theta = torch.atan2(meshy, meshx)

        desired_min, desired_max = -1, 1
        mesh_r = desired_min + ((desired_max-desired_min)/(mesh_r.max()-mesh_r.min()))*(mesh_r-mesh_r.min())
        mesh_theta = desired_min + ((desired_max-desired_min)/(mesh_theta.max()-mesh_theta.min()))*(mesh_theta-mesh_theta.min())

        self.grid = torch.stack((mesh_theta, mesh_r), 2)
    
    def cartesian_proj(self, Fbev):
        # Fbev = (B, C, D, W) 64x32 -> 32x32
        
        Fbev_flipped = torch.flip(Fbev, [2])
        grid = self.grid.unsqueeze(0).repeat(Fbev.shape[0], 1, 1, 1)
        
        bev_proj = torch.nn.functional.grid_sample(Fbev_flipped, grid, mode='bilinear', align_corners=False)
        bev_proj = torch.rot90(bev_proj, dims=[-2,-1])

        return bev_proj
    

class BEV_Layout_Estimation(nn.Module):
    def __init__(self, 
                 backbone_channels=256, 
                 depth_channels=64, 
                 bev_projection_size=32, 
                 classes=8, 
                 device='cpu',
                 return_latent=False):
        
        super().__init__()

        self.BEV_projection = BEV_projection(backbone_channels, depth_channels, bev_projection_size, device=device)
        self.Multi_scale_layout_prediction = MLSP(backbone_channels, classes)        
        
        if return_latent: self.Multi_scale_layout_prediction = nn.Upsample(scale_factor=4, mode='bilinear')            

    def forward(self, x):
        x = self.BEV_projection(x) #cx32x32
        x = self.Multi_scale_layout_prediction(x)
        return x