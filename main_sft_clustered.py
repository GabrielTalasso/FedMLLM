import copy
import os
from tqdm import tqdm
import numpy as np
import time
import json
import gc  # Add garbage collection for memory optimization
#time.sleep(15*60)

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from utils.utils import default_evaluation, save_dataset_test
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

dataset, dataset_test = get_dataset(script_args.dataset_name, script_args.local_data_dir, script_args.train_split)

dataset =      process_sft_dataset(script_args.dataset_name, dataset,      script_args.dataset_sample)
dataset_test = process_sft_dataset(script_args.dataset_name, dataset_test, script_args.dataset_sample)

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset)

if fed_args.evaluation_mode == "local":
    local_datasets_test = split_dataset(fed_args, script_args, dataset_test)

elif fed_args.evaluation_mode == "global":
    aux_split_strategy = fed_args.split_strategy #save the true value
    fed_args.split_strategy = fed_args.split_strategy.split('_')[0] + '_iid' #evaluate with iid data (all domains)
    local_datasets_test = split_dataset(fed_args, script_args, dataset_test)
    fed_args.split_strategy = aux_split_strategy #restore the original value

sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Memory Optimization: Delete original datasets after splitting =====
del dataset, dataset_test
gc.collect()  # Force garbage collection to free memory

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)
print(f"Model loaded from {script_args.model_name_or_path}")

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="left")
if tokenizer.pad_token is None:
    print(f"Pad token is not set, setting it to {tokenizer.unk_token}.")
    tokenizer.pad_token = tokenizer.eos_token

if tokenizer.eos_token == tokenizer.unk_token or tokenizer.pad_token == tokenizer.eos_token:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    print(f"Pad token is set to {tokenizer.pad_token}.")

print('Special tokens:', tokenizer.special_tokens_map)
model.resize_token_embeddings(len(tokenizer))

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
if response_template:
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    packing = False
else:
    data_collator = None
    packing = True

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
idx = None

for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args,script_args,  round)

    if round + 1 == fed_args.sim_round: #return all clients
        clients_this_round = list(range(fed_args.num_clients))

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue

        if round > fed_args.sim_round:
            print(f'Setting parameters for client {client} with cluster {idx[client] - 1}...')
            set_peft_model_state_dict(model, global_dict[idx[client] - 1])

        else:
            print(f'Setting parameters for client {client}...')
            set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
        sub_dataset_test = local_datasets_test[client]
        sub_dataset_test = sub_dataset_test.shuffle(seed=round).select(range(script_args.max_eval_size) if script_args.max_eval_size < len(sub_dataset_test) else range(len(sub_dataset_test)))

        if not os.path.exists(os.path.join(script_args.output_dir, "clients_adapters")):
            os.makedirs(os.path.join(script_args.output_dir, "clients_adapters"))

        if (round+1) in [int(x) for x in fed_args.evaluation_rounds.split(",")]:
            save_dataset_test(sub_dataset_test, script_args, client, round)
            print(f"Evaluating client {client} on the test set with size {len(sub_dataset_test)} for cluster {idx[client] - 1} in round {round}...")
            default_evaluation(
                model=model,
                tokenizer=tokenizer,
                dataset=sub_dataset_test,
                client_id=client,
                round=round,
                formatting_prompts_func=formatting_prompts_func,
                script_args=script_args,
                cluster_id= idx[client] - 1
            )
                        
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-5)      # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)                        # update the training arguments

        # ===== Train local model on the client side =====
        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
            packing=packing,
        )

        # ===== Save initial model adapter in checkpoint-0 =====
        if round == 0 and client == clients_this_round[0]:
            # Save the initial model adapter in checkpoint-0, without training
            trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-0"))

        print(f'Training client {client}...')
        results = trainer.train()
        training_loss[client].append(results.training_loss)

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

        # ===== Save the model =====
        if fed_args.fed_alg == 'MTL':
            trainer.save_model(os.path.join(script_args.output_dir, f"clients_adapters/checkpoint-{round+1}_client{client}"))
        else:
            if (round+1) == fed_args.sim_round:
                trainer.save_model(os.path.join(script_args.output_dir, f"clients_adapters/checkpoint-{round+1}_client{client}"))

    # ===== Server aggregates the local models =====

    if round < fed_args.sim_round:
        global_dict, global_auxiliary = global_aggregate(
            fed_args, script_args, global_dict, local_dict_list, sample_num_list, \
            clients_this_round, round, proxy_dict=proxy_dict, \
            opt_proxy_dict=opt_proxy_dict,
            auxiliary_info=(global_auxiliary, auxiliary_delta_dict),
            round = round
        )

        set_peft_model_state_dict(model, global_dict)   # Update global model

        # ===== Save the model =====
        if (round+1) % fed_args.save_model_freq == 0:
            trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
        if fed_args.fed_alg == 'MTL':
            trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
        
        np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
    
    if round >= fed_args.sim_round:
        global_dict, global_auxiliary, idx = global_aggregate(
            fed_args, script_args, global_dict, local_dict_list, sample_num_list, \
            clients_this_round, round, proxy_dict=proxy_dict, \
            opt_proxy_dict=opt_proxy_dict,
            auxiliary_info=(global_auxiliary, auxiliary_delta_dict),
            round = round, idx = idx
        )

        for cluster in range(fed_args.n_clusters):
            set_peft_model_state_dict(model, global_dict[cluster])
            trainer.save_model(os.path.join(script_args.output_dir, f"cluster_{cluster}_checkpoint-{round+1}"))
    
        np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
