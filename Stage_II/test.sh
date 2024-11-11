export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export CONTROLNET_DIR="./checkpoints/Stage_II/samearea" 
export BEV_CHECKPOINT="./checkpoints/Stage_I/stage_I.pt"
export OUTPUT_DIR="./log_imgs_test"

accelerate launch --multi_gpu ./Stage_II/test.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --stage_I_checkpoint_path=$BEV_CHECKPOINT \
 --output_dir=$OUTPUT_DIR \
 --text_type="dynamic" \
 --VIGOR_mode="samearea" \
 --FOV=360
       


 




