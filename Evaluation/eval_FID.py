import torch
import os
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from SAFA_vgg import SAFA_vgg_sat
import torchvision.transforms as transforms
from PIL import Image

transform = transforms.Compose([transforms.Resize(size=tuple([320,320])),transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])


SAFA_model = SAFA_vgg_sat(safa_heads=8)
SAFA_model = torch.nn.DataParallel(SAFA_model)

missing_keys, unexpected_keys = SAFA_model.load_state_dict(torch.load("PRE-TRAINED_SAFA_WEIGHTS")['model_state_dict'], strict = False)

SAFA_model = SAFA_model.to("cpu")

SAFA_model.eval()

print(missing_keys)

print(unexpected_keys)



test_dict = {
    "EXPERIMENT_NAME":"GENERATED_IMAGES_DIRECTORY",
}

files = []


ref_dir = "GROUND_TRUTH_IMAGES_DIRECTORY"


for name, dirs in test_dict.items():

    print(f"+++++++++++++++++++++ start of {name} +++++++++++++++++++++")

    num_file = sorted(os.listdir(dirs))


    ssim_scores = []
    psnr_scores = []
    mse_scores = []
    lpips_distances = []
    batch_index = 0
    batch_size = 200
    last_batch = False
    fid = FrechetInceptionDistance(feature=SAFA_model.module, reset_real_features=True).to("cuda:0")


    while not last_batch:

        if batch_index + batch_size >= len(num_file):
            batch_size = len(num_file) - batch_index - 1
            last_batch = True

        batch_img = None
        batch_ref_img = None

        for index in range(batch_index, batch_index + batch_size):
            img_name = num_file[index]

            img_dir = os.path.join(dirs, img_name)


            city, image_name = img_name.split("#VAIL#")
            ref_img_dir = os.path.join(ref_dir, city, "grd_aerial", image_name.replace(".jpg", ".png"))

            img = Image.open(img_dir, 'r').convert('RGB')
            ref_img = Image.open(ref_img_dir, 'r').convert('RGB')

            if batch_img == None and batch_ref_img == None:
                batch_img = transform(img).unsqueeze(0).to("cuda:0")
                batch_ref_img = transform(ref_img).unsqueeze(0).to("cuda:0")

            else:
                img = transform(img).unsqueeze(0).to("cuda:0")
                ref_img = transform(ref_img).unsqueeze(0).to("cuda:0")


                batch_img = torch.cat([batch_img, img], axis = 0)
                batch_ref_img = torch.cat([batch_ref_img, ref_img], axis = 0)


        fid.update(batch_ref_img, real=True)
        fid.update(batch_img, real=False)

        batch_index += batch_size

    fid_score = fid.compute()

    print(f"FID score of {name} is {fid_score}")