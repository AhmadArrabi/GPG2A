export OUTPUT_DIR="./Test" 

python ./Stage_I/train.py \
 --batch_size=32 \
 --epoch=40 \
 --output_dir=$OUTPUT_DIR \
 --backbone_channels=256 \
 --depth_channels=64 \
 --bev_projection_size=32 \
 --fov=360