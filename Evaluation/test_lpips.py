import torch
import lpips
import os
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

use_gpu = True         # Whether to use GPU
spatial = False         # Return a spatial map of perceptual distance.

loss_fn = lpips.LPIPS(net='alex', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'

if(use_gpu):
	loss_fn.cuda()

test_dict = {
    "EXPERIMENT_NAME":"GENERATED_IMAGES_DIRECTORY",
}


ref_dir = "GROUND_TRUTH_IMAGES_DIRECTORY"


for name, dirs in test_dict.items():

    num_file = sorted(os.listdir(dirs))

    ssim_scores = []
    psnr_scores = []
    mse_scores = []
    lpips_distances = []
    for i in num_file:
        img_dir = os.path.join(dirs, i)

        city, image_name = i.split("#VAIL#")
        ref_img_dir = os.path.join(ref_dir, city, "grd_aerial", image_name.replace(".jpg", ".png"))

        img = cv2.imread(img_dir)
        ref_img = cv2.imread(ref_img_dir)

        ref_img = cv2.resize(ref_img, (img.shape[0], img.shape[1]), interpolation = cv2.INTER_AREA)

        ex_ref = lpips.im2tensor(ref_img[:,:,::-1]).cuda()
        ex_p0 = lpips.im2tensor(img[:,:,::-1]).cuda()

        lpips_distance = float(loss_fn.forward(ex_ref,ex_p0).detach().cpu())
        ssim_score_0 = ssim(ref_img, img, channel_axis=2)
        psnr_score_0 = psnr(ref_img, img)
        mse_score_0 = mse(ref_img, img)

        ssim_scores.append(ssim_score_0)
        psnr_scores.append(psnr_score_0)
        mse_scores.append(mse_score_0)
        lpips_distances.append(lpips_distance)

    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_mse = sum(mse_scores) / len(mse_scores)
    avg_lpips = sum(lpips_distances) / len(lpips_distances)

    print("+"*30)
    print(f"average ssim for {name} is ", avg_ssim)
    print(f"average psnr for {name} is ", avg_psnr)
    print(f"average mse for {name} is ", avg_mse)
    print(f"average lpips for {name} is ", avg_lpips)
    print("+"*30)