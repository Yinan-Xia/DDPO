set -x
experiment_name='qwen3_dynamic_dyn_avg_th_0.2_dyn_a_1-a_bar_step_105_temp_1_len_25600'
data_path=/share/xiayinan/datasets/gather_data.parquet
save_path='/share/xiayinan/verl/gen_results/GRPO/'${experiment_name}'.parquet'
model_path=/share/xiayinan/verl/checkpoints/GRPO/qwen_14b_limo_grpo_12800_128_accum_1_0_dynamic_dyn_avg_th_0.2_dyn_a_1-a_bar/global_step_35/actor/huggingface
max_prompt_length=1024
max_response_length=20000
python3 -m verl.trainer.main_generation \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=4 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=0 \
    rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    rollout.top_k=-1 \
    rollout.top_p=1 \
    rollout.prompt_length=$max_prompt_length \
    rollout.response_length=$max_response_length \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8 \


