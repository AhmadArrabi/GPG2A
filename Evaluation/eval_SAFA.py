import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import os
import argparse
import json
from SAFA_vgg import SAFA_vgg
import json
import glob

args_do_not_overide = ['data_dir', 'verbose', 'dataset']
def ReadConfig(path):
    all_files = os.listdir(path)
    config_file =  list(filter(lambda x: x.endswith('parameter.json'), all_files))
    with open(os.path.join(path, config_file[0]), 'r') as f:
        p = json.load(f)
        return p


def GetBestModel(path):
    all_files = os.listdir(path)
    if "epoch_last" in all_files:
        all_files.remove("epoch_last")
    config_files =  list(filter(lambda x: x.startswith('epoch_'), all_files))
    config_files = sorted(list(map(lambda x: int(x.split("_")[1]), config_files)), reverse=True)
    best_epoch = config_files[0]
    return os.path.join('epoch_'+str(best_epoch), 'epoch_'+str(best_epoch)+'.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to model weights')
    parser.add_argument('--experiment_name', type=str, help='a preferred experiment name')
    parser.add_argument('--image_path', type=str, help='path to generated images')
    parser.add_argument('--gt_path', type=str, help='path to generated images')

    opt = parser.parse_args()

    config = ReadConfig(opt.model_path)
    for k,v in config.items():
        if k in args_do_not_overide:
            continue
        setattr(opt, k, v)
    
    print(opt)

    number_descriptors = opt.SAFA_head


    SATELLITE_IMG_WIDTH = 320
    SATELLITE_IMG_HEIGHT = 320
    polar_transformation = False
    print("SATELLITE_IMG_WIDTH:",SATELLITE_IMG_WIDTH)
    print("SATELLITE_IMG_HEIGHT:",SATELLITE_IMG_HEIGHT)

    STREET_IMG_WIDTH = 640
    STREET_IMG_HEIGHT = 320

    DESC_LENGTH = 512


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    transforms_sat = transforms.Compose([
        transforms.Resize(size=tuple([SATELLITE_IMG_HEIGHT, SATELLITE_IMG_WIDTH])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
        ])
    
    transforms_grd = transforms.Compose([
        transforms.Resize(size=tuple([STREET_IMG_HEIGHT, STREET_IMG_WIDTH])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
        ])

    model = SAFA_vgg(safa_heads=opt.SAFA_head)
    embedding_dims = number_descriptors * DESC_LENGTH
    
    model = nn.DataParallel(model)
    model.to(device)

    best_model = GetBestModel(opt.model_path)
    best_model = os.path.join(opt.model_path, best_model)
    print("loading model : ", best_model)
    model.load_state_dict(torch.load(best_model)['model_state_dict'])
    model.eval()


    test_dict = {opt.experiment_name : opt.image_path}



    ref_dir = opt.gt_path


    for name, dirs in test_dict.items():

        # num_file = sorted(os.listdir(dirs))
        num_file = sorted(glob.glob(os.path.join(opt.image_path, "*", "*.png")))

        geodtr_same_scores = []
        geodtr_cross_scores = []
        score_dict = {}


        print("start testing...")

        with torch.no_grad():
            for i in tqdm(num_file, disable=False):
                img_dir = os.path.join(dirs, i)

                # city, image_name = i.split("#VAIL#")
                # ref_img_dir = os.path.join(ref_dir, city, "grd_aerial", image_name.replace(".jpg", ".png"))

                # ref_grd_dir = os.path.join(ref_dir, city, "grd_aerial", image_name)

                city_name, image_name = i.split("/")[2], i.split("/")[3]
                ref_img_dir = os.path.join(opt.gt_path, city_name, "grd_aerial", image_name)
                ref_grd_dir = os.path.join(opt.gt_path, city_name, "panorama", image_name.replace(".png", ".jpg"))


                sat_ref = Image.open(ref_img_dir).convert('RGB')
                sat_test = Image.open(img_dir).convert('RGB')
                grd_ref = Image.open(ref_grd_dir).convert('RGB')

                sat_ref = transforms_sat(sat_ref).unsqueeze(0).to(device)
                sat_test = transforms_sat(sat_test).unsqueeze(0).to(device)
                grd_ref = transforms_grd(grd_ref).unsqueeze(0).to(device)

                sat_ref_global, grd_ref_global = model(sat_ref, grd_ref, is_cf=False)
                sat_test_global, grd_ref_global = model(sat_test, grd_ref, is_cf=False)

                sat_ref_global = sat_ref_global.detach().cpu()
                sat_test_global = sat_test_global.detach().cpu()
                grd_ref_global = grd_ref_global.detach().cpu()


                sat_test_global = sat_test_global.permute(1, 0)

                distance_same = torch.matmul(sat_ref_global, sat_test_global)
                distance_same = (2.0 - 2.0 * distance_same) / 4.0

                distance_cross = torch.matmul(grd_ref_global, sat_test_global)
                distance_cross = (2.0 - 2.0 * distance_cross) / 4.0
                
                distance_same = float(distance_same.detach().cpu())
                distance_cross = float(distance_cross.detach().cpu())


                geodtr_same_scores.append(distance_same)
                geodtr_cross_scores.append(distance_cross)

                score_dict[i] = distance_same

        avg_same_geodtr = sum(geodtr_same_scores) / len(geodtr_same_scores)

        avg_cross_geodtr = sum(geodtr_cross_scores) / len(geodtr_cross_scores)

        print("+"*30)
        print(f"average same-view similarity for {name} is ", avg_same_geodtr)
        print(f"average cross-view similarity for {name} is ", avg_cross_geodtr)
        print("+"*30)


    
