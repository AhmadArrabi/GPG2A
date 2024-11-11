import sys
sys.path.insert(1, './')
import torch
import torch.nn as nn
import einops
from models import BEV_Layout_Estimation
from dataset import VIGOR_v2
from segmentation_models_pytorch.losses import DiceLoss
from torchvision.utils import save_image
from utils import map_discrete_to_rgb_semantic_map, map_one_hot_to_rgb_semantic_map
import os
import argparse

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training stage I of GPG2A")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=40
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs",
        help=("the name of the directory that images and checkpoints will be saved in")
    )
    parser.add_argument(
        "--backbone_channels",
        type=int,
        default=256
    )
    parser.add_argument(
        "--depth_channels",
        type=int,
        default=64
    )
    parser.add_argument(
        "--bev_projection_size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--fov",
        type=int,
        default=360
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def main(args):
    ########################### params ##########################
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # logging 
    log_dir = f'{args.output_dir}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f'{log_dir}/log_imgs_train', exist_ok=True)
    os.makedirs(f'{log_dir}/checkpoints', exist_ok=True)

    grd_size = [256, 512]
    layout_size = [128, 128]

    ######################## constants #########################
    color_map = torch.tensor([
            [255., 178., 102.],  #Building
            [64., 90., 255.],    #parking
            [102., 255., 102.],  #grass park playground
            [0., 153., 0.],      #forest
            [204., 229., 255.],  #water
            [192., 192., 192.],  #path
            [96., 96., 96.],     #road
            [255., 255., 255.]   #background
    ], device=DEVICE)

    ##################### data loading #########################
    dataset = VIGOR_v2(mode='train', 
                    split='samearea', 
                    layout_mode='discrete', 
                    FOV=args.fov,
                    grd_size=grd_size, 
                    layout_size=layout_size)
    num_gpus = torch.cuda.device_count()

    ###################### resources #####################
    if num_gpus > 1:
        print(f'USING {num_gpus} GPUS :D')
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=32)
        model = BEV_Layout_Estimation(backbone_channels=args.backbone_channels, depth_channels=args.depth_channels, bev_projection_size=args.bev_projection_size, device=DEVICE)
        model = nn.DataParallel(model)
        model = model.to(DEVICE)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        model = BEV_Layout_Estimation(backbone_channels=args.backbone_channels, depth_channels=args.depth_channels, bev_projection_size=args.bev_projection_size, device=DEVICE).to(DEVICE)
        
    loss = DiceLoss(mode='multiclass', classes=8)
    optimizer = torch.optim.Adam(model.parameters())

    model.train()

    # Training
    print(f'TRAINING SEGMENTATION MODEL WITH {DEVICE} FOR {args.epoch} EPOCHS')
    sys.stdout.flush()

    # logging freq.
    print_freq = int((dataset.__len__() // args.batch_size)*0.2)
    log_freq = int((dataset.__len__() // args.batch_size)*0.2)

    for e in range(args.epoch):
        for i, batch in enumerate(loader):
            I_pano = batch['ground'].to(DEVICE)
            I_layout = batch['layout'].to(DEVICE)
            pred = model(I_pano)

            loss_ = loss(pred, I_layout)
            loss_.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            if not(i%print_freq):
                print(f'Epoch {e} Step{i} - Dice Loss: {loss_}')
                sys.stdout.flush()
            
            if not(i%log_freq):
                chkpt_img = einops.rearrange(map_one_hot_to_rgb_semantic_map(pred, DEVICE), 'b h w c -> b c h w').squeeze()
                save_image(chkpt_img/255, f'{log_dir}/log_imgs_train/FAKE_e{e}_s{i}.jpg')
                chkpt_layout = map_discrete_to_rgb_semantic_map(I_layout, color_map)
                save_image(chkpt_layout/255, f'{log_dir}/log_imgs_train/REAL_e{e}_s{i}.jpg')
                
        print('finished epoch {}'.format(e))
        sys.stdout.flush()
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_}, f'{log_dir}/checkpoints/checkpoint_epoch_{e}.pt')
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
