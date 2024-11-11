***This folder contains the evaluation code for GPG2A Ground-to-Aerial (G2A) image synthesis model.***

### Prerequisite
```
pytorch
torchvision
scikit-image
opencv
lpips
PIL
tqdm
```

### Pre-trained SAFA model
Please download the pretrained SAFA model [here](https://drive.google.com/file/d/1z6BB_CUQxDyN4y7LUbxhJcoh75f9MW5N/view?usp=sharing) and extract to the same directory.

### Generate aerial images
Please follow the training and testing tutorial to generate aerial images for evaluation with correct naming and directory.

### $Sim_c$ and $Sim_s$
1. Please replace the `GROUND_TRUTH_IMAGES_DIRECTORY` in line 92 of `eval_SAFA.py` with your downloaded VIGORv2 dataset directory.
2. Please add the generated images in line `test_dict`(line 88 in `eval_SAFA.py`) with a preferred experiment name.
3. To calculate $Sim_c$ and $Sim_s$, please run
```python eval_SAFA.py --model_path SAFA_PRETREAINED_MODEL_PATH```.

### FID-SAFA
1. Please replace `PRE-TRAINED_SAFA_WEIGHTS` in line 15 of `eval_FID.py` with your download SAFA pretrained weights directory (i.e. `./GPG2A_eval_SAFA_model/epoch_100.pth`).
2. Please replace the `GROUND_TRUTH_IMAGES_DIRECTORY` in line 34 of `eval_FID.py` with your downloaded VIGORv2 dataset directory.
3. Please add the generated images in line `test_dict`(line 27 in `eval_FID.py`) with a preferred experiment name.
4. Running `python eval_FID.py`.

### LPIPS, SSIM, PSNR
1. Please replace the `GROUND_TRUTH_IMAGES_DIRECTORY` in line 22 of `test_lpips.py` with your downloaded VIGORv2 dataset directory.
2. Please add the generated images in line `test_dict`(line 17 in `test_lpips.py`) with a preferred experiment name.
3. Please run
```python test_lpips.py```.