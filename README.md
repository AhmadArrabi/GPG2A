# GPG2A
Official code repository for the paper "[Cross-View Meets Diffusion: Aerial Image Synthesis with Geometry and Text Guidance](https://arxiv.org/abs/2408.04224)" presented at WACV, 2025.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)

![SVG Image](Assets/model.SVG)

## Prerequisites
- accelerate
- diffusers
- Pytorch >= 2.1.2
- torchvision >= 0.16.2
- pillow
- tqdm

## Dataset
VIGORv2 can be downloaded from the following link.

## Pre-trained Weights
Our checkpoints can be downloaded from the following link.

## Training
After downloading VIGORv2 and the pre-trained weights, for the best experience, make sure your repository tree is as follows:
```bash
GPG2A
├── checkpoints
│   ├── Stage_I/
│   └── Stage_II/
├── data
│   └── VIGORv2/
├── Stage_I/
├── Stage_II/
└── utils.py
```

### Stage I (BEV Layout Estimation)
To train stage I (BEV Layout Estimation) navigate to the root directory and run the following script
```
bash ./Stage_I/train.sh
```
Note that you can customize the training experiment using the arguments provided in `./Stage_I/train.py` by simply passing them to `./Stage_I/train.sh`

This script will save checkpoints at every epoch trained in `./OUTPUT_DIR/checkpoints`

### Stage II (Diffusion Aerial Synthesis)
To train stage II (Diffusion Aerial Synthesis) navigate to the root directory and run the following script
```
bash ./Stage_II/train.sh
```
Note that you need to have the checkpoints of stage I saved in the correct location (as shown in the above tree).
You can customize the training experiment using the arguments provided in `./Stage_II/train.py` by simply passing them to `./Stage_II/train.sh`, for example:
```
--text_type="dynamic"
--VIGOR_mode="samearea"
```
The following are the main experiments ran in the paper:
| Argument       | Experiment    |
|----------------|---------------|
| `--text_type="dynamic"` | Dynamic text prompt extracted from Gemini discriptions of the ground image |
| `--text_type="raw"` | Gemini descriptions of the ground image without processing |
| `--text_type="constant"` | A constant single generic prompt, e.g., "High-quality aerial image" |
| `--text_type="city"` | Same as 'constant' prompt but with city name |
| `--text_type="empty"` | Empty string |
| `--VIGOR_mode="samearea"` | Same-area training protocol |
| `--VIGOR_mode="crossarea"` | Cross-area training protocol|
| `--f_bev` | Use the raw BEV feature instead of the estimated layout map |

## Testing
### Stage I (BEV Layout Estimation)
To test stage I (BEV Layout Estimation) navigate to the root directory and run the following script
```
bash ./Stage_I/test.sh
```
Note that you need to have the checkpoints of stage I saved in the correct location (as shown in the above tree).

### Stage II (Diffusion Aerial Synthesis)
To test stage II (Diffusion Aerial Synthesis) navigate to the root directory and run the following script
```
bash ./Stage_II/test.sh
```
Note that you need to have the checkpoints of stage I and II saved in the correct location (as shown in the above tree).

This script saves all generated images of the test set in the specified output directory (`./log_imgs_test` by default)



