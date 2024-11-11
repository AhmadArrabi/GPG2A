export MODEL_DIR="runwayml/stable-diffusion-v1-5" 
export OUTPUT_DIR="./Test" 
export BEV_CHECKPOINT="./checkpoints/Stage_I/stage_I.pt"

accelerate launch --multi_gpu ./Stage_II/train.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --stage_I_checkpoint_path=$BEV_CHECKPOINT \
 --resolution=256 \
 --learning_rate=1e-5 \
 --train_batch_size=32 \
 --num_train_epochs=20 \
 --checkpointing_steps=2500 \
 --text_type="dynamic" \
 --VIGOR_mode="samearea" 
 





 




