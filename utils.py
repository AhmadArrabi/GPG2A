import torch
from torchvision import transforms
import math

color_dict = {
    (255., 178., 102.): 0,  #Building
    (64., 90., 255.): 1,    #parking
    (102., 255., 102.): 2,  #grass park playground
    (0., 153., 0.): 3,      #forest
    (204., 229., 255.): 4,  #water
    (192., 192., 192.): 5,  #path
    (96., 96., 96.): 6,     #road
    (255., 255., 255.): 7   #background
}

def map_one_hot_to_rgb_semantic_map(one_hot, device='cuda'):
    colors = torch.tensor(list(color_dict.keys()), device=device)
    indices = torch.argmax(one_hot, axis=1)
    rgb_semantic_map = colors[indices]
    return rgb_semantic_map

def map_discrete_to_rgb_semantic_map(input, color_tensor):
    #return color_tensor[input].permute(2,0,1)
    return color_tensor[input].permute(0,3,1,2)

def HFlip(img):
    return torch.flip(img, [2])

def Rotate(sat, grd, orientation, is_polar=False):
    height, width = grd.shape[1], grd.shape[2]
    if orientation == 'left':
        if is_polar:
            left_sat = sat[:, :, 0:int(math.ceil(width * 0.75))]
            right_sat = sat[:, :, int(math.ceil(width * 0.75)):]
            sat_rotate = torch.cat([right_sat, left_sat], dim=2)
        else:
            sat_rotate = torch.rot90(sat, -1, [1, 2])
        left_grd = grd[:, :, 0:int(math.ceil(width * 0.75))]
        right_grd = grd[:, :, int(math.ceil(width * 0.75)):]
        grd_rotate = torch.cat([right_grd, left_grd], dim=2)

    elif orientation == 'right':
        if is_polar:
            left_sat = sat[:, :, 0:int(math.floor(width * 0.25))]
            right_sat = sat[:, :, int(math.floor(width * 0.25)):]
            sat_rotate = torch.cat([right_sat, left_sat], dim=2)
        else:
            sat_rotate = torch.rot90(sat, 1, [1,2])
        left_grd = grd[:, :, 0:int(math.floor(width * 0.25))]
        right_grd = grd[:, :, int(math.floor(width * 0.25)):]
        grd_rotate = torch.cat([right_grd, left_grd], dim=2)

    elif orientation == 'back':
        if is_polar:
            left_sat = sat[:, :, 0:int(width * 0.5)]
            right_sat = sat[:, :, int(width * 0.5):]
            sat_rotate = torch.cat([right_sat, left_sat], dim=2)
        else:
            sat_rotate = torch.rot90(sat, 1, [1,2])
            sat_rotate = torch.rot90(sat_rotate, 1, [1,2])
        left_grd = grd[:, :, 0:int(width * 0.5)]
        right_grd = grd[:, :, int(width * 0.5):]
        grd_rotate = torch.cat([right_grd, left_grd], dim=2)
    
    else:
        raise RuntimeError(f"Orientation {orientation} is not implemented")

    return sat_rotate, grd_rotate

def layout_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor()
    ])

def aerial_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5,.5,.5],
                   std=[.5,.5,.5])
    ])

def ground_transform(size, augmentation=True):
    if augmentation:
        return transforms.Compose([
            transforms.Resize(size=tuple(size)),
            transforms.ColorJitter(0.15, 0.15, 0.15),
            transforms.RandomGrayscale(p=0.15),
            transforms.RandomPosterize(p=0.15, bits=4),
            transforms.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 5)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=tuple(size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
    
