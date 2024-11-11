import torch
from PIL import Image
import os
import csv 
import random
from utils import *
from random import randrange

class VIGOR_v2(torch.utils.data.Dataset):
    """
    VIGOR dataset for all 4 cities [NewYork, Chicago, Seattle, SanFrancisco]
    args:
        - root: data path 
        - mode: choose ["train", "test"]
        - split: choose ["samearea", "crossarea"]
        - text_type: choose ["constant", "city", "raw", "dynamic", "empty"]
        - FOV: continuous value between (0,360)
        - aerial_size: [height, width]
        - grd_size: [height, width]
        - augmentation: Bool value indicating augmentation
        - cities: list of cities to include
    """
    def __init__(self, 
                 root ='./data/VIGORv2', 
                 mode="train",
                 split="samearea",
                 text_type="constant",
                 FOV=360,
                 aerial_size=[256,256], 
                 grd_size=[1024,2048], 
                 augmentation=True,
                 cities=['NewYork', 'SanFrancisco', 'Chicago', 'Seattle'],
                 args=None,):
        
        super(VIGOR_v2, self).__init__()

        self.args = args
        self.root = root
        self.mode = mode
        self.FOV = FOV
        self.text_type = text_type
        self.augmentation = augmentation

        self.aerial_size = aerial_size
        self.grd_size = grd_size

        self.transform_ground = ground_transform(size=self.grd_size, augmentation=self.augmentation)
        self.transform_aerial = aerial_transform(size=self.aerial_size)

        self.list = []

        if text_type=="raw":
            data_csv_file_name = f"VIGOR_geosplit_{mode}_processed.csv"
        else:
            data_csv_file_name = f"VIGOR_geosplit_{mode}_processed_dynamic.csv"

        if split == 'samearea':
            if mode == "train":
                with open(f'{self.root}/{data_csv_file_name}', 'r') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    for row in csv_reader:
                        if row[2] in cities:
                            self.list.append((row[2], row[3], row[4])) #(city, image_name, prompt) e.g. ('Seattle', 'Xokod_2342kfwmeokf.png', 'highquality aerial ..)
            elif mode == "test":
                with open(f'{self.root}/{data_csv_file_name}', 'r') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    for row in csv_reader:
                        if row[2] in cities:
                            self.list.append((row[2], row[3], row[4]))
            else:
                raise RuntimeError(f"{mode} is not implemented")
        
        elif split == 'crossarea':
            if mode == "train":
                with open(f'{self.root}/{data_csv_file_name}', 'r') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    for row in csv_reader:
                        if row[2]=='Seattle' or row[2]=='NewYork':
                            self.list.append((row[2], row[3], row[4]))
                
                with open(f'{self.root}/{data_csv_file_name}', 'r') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    for row in csv_reader:
                        if row[2]=='Seattle' or row[2]=='NewYork':
                            self.list.append((row[2], row[3], row[4]))
            
            elif mode == "test":
                with open(f'{self.root}/{data_csv_file_name}', 'r') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    for row in csv_reader:
                        if row[2]=='Chicago' or row[2]=='SanFrancisco':
                            self.list.append((row[2], row[3], row[4]))
            else:
                raise RuntimeError(f"{mode} is not implemented")
        else:
            raise RuntimeError(f"{split} is not implemented")        

        print("Number of aerial-ground pairs : ", len(self.list))

    def __getitem__(self, index):
        image_name = self.list[index]

        ground = Image.open(os.path.join(self.root, image_name[0], 'panorama', image_name[1].replace('.png', '.jpg')), 'r').convert('RGB')
        width, height = ground.size
        ground = ground.crop((0, height / 4, width,  3 * height / 4))
        ground = self.transform_ground(ground)

        if self.FOV != 360:
            occlusion_degree = int(360-self.FOV)
            occlusion_box_width = (self.grd_size[1]*occlusion_degree)//(360)
            occlusion_box_center = randrange(occlusion_box_width//2 + 1, self.grd_size[1]-(occlusion_box_width//2)-1)
            ground[:,:,occlusion_box_center-occlusion_box_width//2:occlusion_box_center+occlusion_box_width//2] = 1.

        aerial = Image.open(os.path.join(self.root, image_name[0], 'grd_aerial', image_name[1]), 'r').convert('RGB')
        aerial = self.transform_aerial(aerial)

        if self.augmentation:
            hflip = random.randint(0,1)
            if hflip == 1:
                aerial = HFlip(aerial)
                ground = HFlip(ground)
            else:
                pass

            orientation = random.choice(["left", "right", "back", "none"])
            if orientation == "none":
                pass
            else:
                aerial, ground = Rotate(aerial, ground, orientation)
        
        if self.text_type == "constant":
            prompt = "Realistic aerial satellite top view image with high quality details with buildings and roads"
        elif self.text_type == "city":
            prompt = f"Realistic {image_name[0]} aerial satellite top view image with high quality details with buildings and roads in {image_name[0]}"
        elif self.text_type == "dynamic" or self.text_type == "raw":
            prompt = image_name[2]
        elif self.text_type == "empty":
            prompt = ""
        else:
            raise RuntimeError(f"{self.text_type} text type is not implemented")    

        return dict(pixel_values=aerial.float(), input_ids=prompt, ground=ground, img_name=image_name)
        
    def __len__(self):
        return len(self.list)
