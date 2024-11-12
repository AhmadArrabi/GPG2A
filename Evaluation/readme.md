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
Please download the pretrained SAFA model [here](https://drive.google.com/file/d/1z6BB_CUQxDyN4y7LUbxhJcoh75f9MW5N/view?usp=sharing) and extract to the root directory.

### Generate aerial images
Please follow the training and testing tutorial to generate aerial images for evaluation with correct naming and directory.

### $Sim_c$ and $Sim_s$
To calculate $Sim_c$ and $Sim_s$, please run the command below,

```python eval_SAFA.py --model_path SAFA_PRETREAINED_MODEL_PATH --experiment_name YOUR_PREFERRED_EXPERIMENT_NAME --image_path GENERATED_IMAGE_PATH --gt_path GROUND_TRUTH_IMAGE_PATH```

For example,

```python eval_SAFA.py --model_path ../GPG2A_eval_SAFA_model/ --experiment_name GPG2A_experiment --image_path ../log_imgs_test/ --gt_path ../Data/VIGOR/```

### FID-SAFA
To calculate FID-SAFA, please run the command below,

```python eval_FID.py --model_path SAFA_PRETREAINED_MODEL_PATH --experiment_name YOUR_PREFERRED_EXPERIMENT_NAME --image_path GENERATED_IMAGE_PATH --gt_path GROUND_TRUTH_IMAGE_PATH```

For example,

```python eval_FID.py --model_path ../GPG2A_eval_SAFA_model/ --experiment_name GPG2A_experiment --image_path ../log_imgs_test/ --gt_path ../Data/VIGOR/```

### LPIPS, SSIM, PSNR
To calculate LPIPS, SSIM, PSNR, please run the command below,

```python test_lpips.py --experiment_name YOUR_PREFERRED_EXPERIMENT_NAME --image_path GENERATED_IMAGE_PATH --gt_path GROUND_TRUTH_IMAGE_PATH```

For example,

```python test_lpips.py --experiment_name GPG2A_experiment --image_path ../log_imgs_test/ --gt_path ../Data/VIGOR/```