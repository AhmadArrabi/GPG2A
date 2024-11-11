export OUTPUT_DIR="./Test" 
export STAGE_I_PATH="./checkpoints/Stage_I/stage_I.pt"

python ./Stage_I/test.py \
 --batch_size=32 \
 --output_dir=$OUTPUT_DIR \
 --stage_I_checkpoint_path=$STAGE_I_PATH \
 --backbone_channels=256 \
 --depth_channels=64 \
 --bev_projection_size=32 \
 --fov=360