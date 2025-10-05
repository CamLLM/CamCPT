export MEGATRON_LM_PATH='Megatron-LM/Megatron-LM-core_v0.13.0'
model_path="Qwen3-8B-Base-mcore"
dataset_path="Mbook_pt.jsonl"
output_dir="megatron_output/Qwen3-8B-Base/Mbook"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron pt \
    --load $model_path \
    --dataset $dataset_path \
    --tensor_model_parallel_size 4 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --bf16 true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 4 \
    --max_epochs 5 \
    --save-interval 128 \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save $output_dir \
    --log_interval 1 \
    --max_length 8192 \
    --num_workers 4 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash \
    --optimizer_cpu_offload true \
    --use-precision-aware-optimizer true \
    --optimizer_offload_fraction 1
