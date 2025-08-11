max_steps=10
num_train_epochs=1
num_rounds=100
eval_round="10,25,50,75,100"
batch_size=16
batch_size_eval=128
gradient_accumulation_steps=1
seq_length=1024
num_clients=8
sample_clients=8
lora_r=8
lora_alpha=16  # twice of lora_r
lr=5e-4

# local_data_dir=""       # you may uncomment this line if your data is stored locally and include it in the python command

dataset_name='CohereForAI/aya_dataset'
#dataset_name="multi_language_clusters"

output_dir="output/experiments"

dataset_sample=400000

sim_alias='clustered'

model_name_or_path='HuggingFaceTB/SmolLM-360M'

gpu='0'
fed_alg="clustered"

CUDA_VISIBLE_DEVICES=$gpu python main_sft_clustered.py \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_name \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --max_steps $max_steps \
 --num_train_epochs $num_train_epochs \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --peft_lora_r $lora_r \
 --peft_lora_alpha $lora_alpha \
 --use_peft True \
 --load_in_4bit True \
 --output_dir $output_dir \
 --template "alpaca" \
 --sim_round 1 \
 --n_clusters 5 \
 --split_strategy "language_clusters" \
 --train_split 0.8 \
 --sim_alias $sim_alias \
 --evaluation_rounds $eval_round \
 --eval_batch_size $batch_size_eval \
 --evaluation_mode "local" \