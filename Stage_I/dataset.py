import torch
from PIL import Image
import os
import csv 
import random
from utils import *
from random import randrange
import einops

class VIGOR_v2(torch.utils.data.Dataset):
    """
    VIGOR dataset for all 4 cities [NewYork, Chicago, Seattle, SanFrancisco]
    args:
        - root: data path
        - mode: choose ["train", "test"]
        - split: choose ["samearea", "crossarea"]
        - FOV: continuous value between (0,360)
        - augmentation: Bool value indicating augmentation
        - layout_size: [height, width]
        - grd_size: [height, width]
        - cities: list of cities to include
        - layout_mode" choose ["RGB", "discrete", "one_hot"]
    """
    def __init__(self, 
                 root ='./data/VIGOR', 
                 mode="train",
                 split="samearea",
                 FOV=360,
                 augmentation=True,
                 layout_size=[256,256], 
                 grd_size=[1024,2048],
                 cities=['NewYork', 'SanFrancisco', 'Chicago', 'Seattle'], 
                 layout_mode='RGB',
                 args=None):
        
        super(VIGOR_v2, self).__init__()

        self.args = args
        self.root = root
        self.mode = mode
        self.FOV = FOV
        self.grd_size = grd_size
        self.layout_size = layout_size
        self.augmentation = augmentation
        self.layout_mode = layout_mode
        
        self.transform_ground = ground_transform(size=self.grd_size, augmentation=self.augmentation)
        self.transform_layout = layout_transform(size=self.layout_size)

        self.list = []

        data_csv_file_name = f"VIGOR_geosplit_{mode}_processed_dynamic.csv"

        if split == 'samearea':
            if mode == "train":
                with open(f'{self.root}/{data_csv_file_name}', 'r') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    for row in csv_reader:
                        if row[2] in cities:
                            self.list.append((row[2], row[3])) #(city, image_name) e.g. ('Seattle', 'Xokod_2342kfwmeokf.png')
            elif mode == "test":
                with open(f'{self.root}/{data_csv_file_name}', 'r') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    for row in csv_reader:
                        if row[2] in cities:
                            self.list.append((row[2], row[3]))
            else:
                raise RuntimeError(f"{mode} is not implemented")
        
        elif split == 'crossarea':
            if mode == "train":
                with open(f'{self.root}/{data_csv_file_name}', 'r') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    for row in csv_reader:
                        if row[2]=='Seattle' or row[2]=='NewYork':
                            self.list.append((row[2], row[3]))
                
                with open(f'{self.root}/{data_csv_file_name}', 'r') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    for row in csv_reader:
                        if row[2]=='Seattle' or row[2]=='NewYork':
                            self.list.append((row[2], row[3]))
            
            elif mode == "test":
                with open(f'{self.root}/{data_csv_file_name}', 'r') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    for row in csv_reader:
                        if row[2]=='Chicago' or row[2]=='SanFrancisco':
                            self.list.append((row[2], row[3]))
            else:
                raise RuntimeError(f"{mode} is not implemented")
        else:
            raise RuntimeError(f"{split} is not implemented")        

        print("Number of layout-ground pairs : ", len(self.list))

    def __getitem__(self, index):
        image_name = self.list[index]

        layout = Image.open(os.path.join(self.root, image_name[0], 'grd_layout', image_name[1]), 'r').convert('RGB')    
        layout = self.transform_layout(layout)

        ground = Image.open(os.path.join(self.root, image_name[0], 'panorama', image_name[1].replace('.png', '.jpg')), 'r').convert('RGB')
        width, height = ground.size
        ground = ground.crop((0, height / 4, width,  3 * height / 4))
        ground = self.transform_ground(ground)

        if self.FOV != 360:
            occlusion_degree = int(360-self.FOV)
            occlusion_box_width = (self.grd_size[1]*occlusion_degree)//(360)
            occlusion_box_center = randrange(occlusion_box_width//2 + 1, self.grd_size[1]-(occlusion_box_width//2)-1)
            ground[:,:,occlusion_box_center-occlusion_box_width//2:occlusion_box_center+occlusion_box_width//2] = 1.

        if self.augmentation:
            hflip = random.randint(0,1)
            if hflip == 1:
                layout = HFlip(layout)
                ground = HFlip(ground)
            else:
                pass

            orientation = random.choice(["left", "right", "back", "none"])
            if orientation == "none":
                pass
            else:
                layout, ground = Rotate(layout, ground, orientation)
        
        if self.layout_mode == 'RGB':
            pass
        elif self.layout_mode == 'discrete':
            layout = self.decode_layout(layout)
        elif self.layout_mode == 'one_hot':
            layout = self.decode_layout(layout)
        else:
            raise RuntimeError(f"layout mode {self.layout_mode} invalid please choose [RGB, discrete, one_hot]")

        return dict(layout=layout, ground=ground, img_name=image_name[1])

    def __len__(self):
        return len(self.list)
    
    def decode_layout(self, image):
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

        classes = torch.Tensor(list(color_dict.values()))
        color_to_class_tensor = torch.Tensor(list(color_dict.keys()))/255

        image = einops.rearrange(image, 'c h w -> h w c')
        indices = torch.argmin(torch.sum(torch.abs(image.unsqueeze(dim=-2) - color_to_class_tensor), dim=-1), dim=-1)
        one_hot = torch.nn.functional.one_hot(indices, num_classes=len(classes))
        one_hot = einops.rearrange(one_hot, 'h w c -> c h w')
        
        if self.layout_mode == 'discrete':
            one_hot = torch.argmax(one_hot, dim=0)
            return one_hot
        elif self.layout_mode == 'one_hot':
            return one_hot

