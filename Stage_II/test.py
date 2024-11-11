import sys
sys.path.insert(1, './')
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from dataset import VIGOR_v2
from Stage_I.models import BEV_Layout_Estimation
from utils import *
import einops
import argparse
import os

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Test script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--f_bev",
        action="store_true",
        default=False,
        help=(
            "Choose to condition on raw latent BEV feature (fbev), this was used in an ablation study"
            ),
    )
    parser.add_argument(
        "--VIGOR_mode",
        type=str,
        default="samearea",
        help=(
            "choose 'samearea' or 'crossarea' for VIGOR dataset"
        ),
    )
    parser.add_argument(
        "--FOV",
        type=int,
        default=360,
        help=(
            "choose 360, 270, 180, 90"
        ),
    )
    parser.add_argument(
        "--text_type",
        type=str,
        default="constant",
        help=(
            "text type for training, choose ['constant', 'city', 'raw', 'dynamic', 'empty']"
        ),
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for the testing dataloader."
    )
    parser.add_argument(
        "--stage_I_checkpoint_path",
        type=str,
        default="constant",
        help=(
            "Path of stage I checkpoint .pt file"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    generator = torch.manual_seed(0)

    #os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f'{args.output_dir}/Chicago', exist_ok=True)
    os.makedirs(f'{args.output_dir}/NewYork', exist_ok=True)
    os.makedirs(f'{args.output_dir}/SanFrancisco', exist_ok=True)
    os.makedirs(f'{args.output_dir}/Seattle', exist_ok=True)
    
    conditioning_channel_dim = 256 if args.f_bev else 3
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, torch_dtype=torch.float16,  conditioning_channels=conditioning_channel_dim)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(args.pretrained_model_name_or_path, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None)

    # Stage I model init
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BEV_model = BEV_Layout_Estimation(device=DEVICE, return_latent=args.f_bev).to(DEVICE)
    BEV_chkpt = torch.load(f'{args.stage_I_checkpoint_path}')
    BEV_model.load_state_dict(BEV_chkpt['model_state_dict'], strict=False)
    BEV_model.eval()
    BEV_model.requires_grad_(False)

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # memory optimization.
    pipe.enable_model_cpu_offload()

    test_dataset = VIGOR_v2(mode='test', split=args.VIGOR_mode, text_type=args.text_type, FOV=args.FOV, aerial_size=[256,256], grd_size=[256,512], augmentation=False)
    
    dataloader_num_workers=8
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=dataloader_num_workers,
        )
    
    for step, batch in enumerate(test_dataloader):
        prompt = batch['input_ids']
        pred = BEV_model(batch['ground'].to(DEVICE))
        
        if args.f_bev:
            control_image = (torch.nn.Upsample(scale_factor=2, mode='bilinear')(pred)).float()
        else:
            pred = einops.rearrange(map_one_hot_to_rgb_semantic_map(pred), 'b h w c -> b c h w').squeeze()
            control_image = ((torch.nn.Upsample(scale_factor=2, mode='bilinear')(pred))/255).float()
        
        model_output = pipe(
            prompt, num_inference_steps=20, generator=generator, image=control_image
        )

        images = model_output.images
        image_name = batch['img_name']

        for i, image in enumerate(images):
            image.save(f'{args.output_dir}/{image_name[0][i]}/{image_name[1][i]}')
            

if __name__ == "__main__":
    args = parse_args()
    main(args)


