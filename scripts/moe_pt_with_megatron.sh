export MEGATRON_LM_PATH='Megatron-LM/Megatron-LM-core_v0.13.0'

model_path="Qwen3-30B-A3B-Base-mcore"
dataset_path="Mbook_pt.jsonl"
output_dir="megatron_output/Qwen3-30B-A3B-Base/Mbook"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron pt \
    --load $model_path \
    --dataset $dataset_path \
    --expert_model_parallel_size 4 \
    --tensor_model_parallel_size 1 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 4 \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save $output_dir \
    --save_interval 1000 \
    --log_interval 1 \
    --max_length 8192 \
    --max_epochs 3 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer true \
    --optimizer_offload_fraction 1 \
    --attention_backend flash \
    --log_memory_to_tensorboard true \
    --tensorboard_log_interval 1
