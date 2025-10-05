export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
model_path="Qwen3-8B-Base"
dataset_path="Mbook_pt.jsonl"
output_dir="swift_output/Qwen3-8B-Base/Mbook"
nproc_per_node=8
NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift pt \
    --model $model_path \
    --train_type full \
    --dataset $dataset_path \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --save_strategy epoch \
    --save_total_limit 10 \
    --deepspeed zero3 \
    --gradient_checkpointing true \
    --use_liger_kernel true \
    --max_length 8192 \
    --num_train_epochs 10 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --save_only_model true \
    --output_dir $output_dir \
    --attn_impl flash_attn \
    --report_to tensorboard
