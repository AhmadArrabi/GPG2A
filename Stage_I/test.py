import sys
sys.path.insert(1, './')
import torch
import torch.nn as nn
import einops
from models import BEV_Layout_Estimation
from dataset import VIGOR_v2
from segmentation_models_pytorch.losses import *
import segmentation_models_pytorch as smp
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
        "--root",
        type=str,
        default="./"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs",
        help=("the name of the directory that images will be saved in")
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
    parser.add_argument(
        "--stage_I_checkpoint_path",
        type=str,
        required=True
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def main(args):

    ########################### params ##########################
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    log_dir = f'{args.output_dir}'
    os.makedirs(log_dir, exist_ok=True)

    grd_size = [256,512]
    layout_size = [128,128]

    ######################## constants #########################
    color_map = torch.tensor([
            [255., 178., 102.],  #Building
            [64., 90., 255.],    #parking
            [102., 255., 102.],  #grass park playground
            [0., 153., 0.],      #forest
            [204., 229., 255.],  #water
            [192., 192., 192.],  #path
            [96., 96., 96.],     #road
            [255., 255., 255.]  #background
    ], device=DEVICE)
    
    ##################### data loading #########################
    dataset = VIGOR_v2(mode='test', 
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
        model = BEV_Layout_Estimation(backbone_channels=args.backbone_channels, depth_channels=args.depth_channels, device=DEVICE, bev_projection_size=args.bev_projection_size)
        model = nn.DataParallel(model)
        model = model.to(DEVICE)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        model = BEV_Layout_Estimation(backbone_channels=args.backbone_channels, depth_channels=args.depth_channels, device=DEVICE, bev_projection_size=args.bev_projection_size).to(DEVICE)

    chkpt = torch.load(f'{args.stage_I_checkpoint_path}')

    model.load_state_dict(chkpt['model_state_dict'])
    
    loss = DiceLoss(mode='multiclass', classes=8)
    model.eval()

    print(f'TESTING SEGMENATION MODEL WITH {DEVICE}')
    sys.stdout.flush()

    # logging freq.
    log_freq = int((dataset.__len__() // args.batch_size)*0.2)

    with torch.no_grad():
        iou_score_micro = []
        iou_score_macro = []
        f1_score_micro  = []
        f1_score_macro  = []

        print('iou_micro iou_macro f1_micro f1_macro Jaccard')
        sys.stdout.flush()

        for i, batch in enumerate(loader):
            I_ground = batch['ground'].to(DEVICE)
            I_layout = batch['layout'].to(DEVICE)
            pred = model(I_ground)
            loss_ = loss(pred, I_layout)

            if not(i%log_freq):
                chkpt_img = einops.rearrange(map_one_hot_to_rgb_semantic_map(pred), 'b h w c -> b c h w').squeeze()
                save_image(chkpt_img/255, f'{log_dir}/FAKE_s{i}.jpg')
                chkpt_layout = map_discrete_to_rgb_semantic_map(I_layout, color_map)
                save_image(chkpt_layout/255, f'{log_dir}/REAL_s{i}.jpg')
            
            pred = torch.argmax(pred, 1)

            tp, fp, fn, tn = smp.metrics.functional.get_stats(pred, I_layout, mode='multiclass', num_classes=8)

            iou_micro = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            iou_macro = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
            f1_micro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            f1_macro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")

            print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(iou_micro, iou_macro, f1_micro, f1_macro, loss_))
            sys.stdout.flush()

            iou_score_micro.append(iou_micro)
            iou_score_macro.append(iou_macro)
            f1_score_micro.append(f1_micro)
            f1_score_macro.append(f1_macro)

        iou_score_micro = torch.stack(iou_score_micro).mean()
        iou_score_macro = torch.stack(iou_score_macro).mean()
        f1_score_micro  = torch.stack(f1_score_micro).mean()
        f1_score_macro  = torch.stack(f1_score_macro).mean()

        print('-'*50)
        sys.stdout.flush()
        print('iou_micro iou_macro f1_micro f1_macro\n{:.4f} {:.4f} {:.4f} {:.4f}'.format(iou_micro, iou_macro, f1_micro, f1_macro))
        sys.stdout.flush()
    
if __name__ == '__main__':
    args = parse_args()
    main(args)