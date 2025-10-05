export MEGATRON_LM_PATH='Megatron-LM-core_v0.13.0'
# Dense, MoE model path
model_path=""
output_dir=""
swift export \
    --model $model_path \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir $output_dir \
    --test_convert_precision true
