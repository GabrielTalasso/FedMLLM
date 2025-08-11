import sys
import os
from tqdm import tqdm
import numpy as np
import torch
sys.path.append(".")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from utils.template import TEMPLATE_DICT
import json
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

template = TEMPLATE_DICT['alpaca'][0]
MODEL_NAME = 'TinyLlama/TinyLlama_v1.1'
DATASET_NAME = "CohereForAI/aya_dataset"
DEVICE = 'cuda:0'
EVALSET_LEN = 10000

def load_model(path, MODEL_NAME, DEVICE):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                                                    quantization_config = BitsAndBytesConfig(
                                                                            load_in_4bit=True,
                                                                            bnb_4bit_use_double_quant=True,
                                                                            bnb_4bit_quant_type="nf4",
                                                                            bnb_4bit_compute_dtype=torch.bfloat16,
                                                                        ),
                                                    device_map={"": Accelerator().local_process_index})

    model = PeftModel.from_pretrained(model, path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device=DEVICE, use_fast=False, padding_side="left")
    tokenizer.pad_token = tokenizer.unk_token

    return model, tokenizer

def load_eval_data(DATASET_NAME, EVALSET_LEN, category, category_name = 'language'):
    
    dataset = load_dataset(DATASET_NAME, split="train", )
    if category_name == 'language':
        dataset = dataset.filter(lambda x: x[category_name] in ['English', 'Swedish', 'German', 'Portuguese', 'Spanish'])
    dataset_splited = dataset.train_test_split(test_size= 0.2, seed=0)
    dataset_test = dataset_splited['test']
    dataset = dataset_test.filter(lambda x: x[category_name] in category)
    dataset_len = min(len(dataset), EVALSET_LEN)
    dataset = dataset.select(range(dataset_len))

    return dataset

def format_instruction(instruction, response, eos):
    template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}"""

    return template.format(instruction, response, eos)

def calculate_perplexity(model, tokenizer, dataset, max_length=512, exp_type = 'instruction'):
    model.eval()
    total_loss = 0
    total_length = 0
    losses = []

    with torch.no_grad():
        for item in tqdm(dataset):
            # Format the input as an instruction
            if exp_type == 'instruction':
             
                input_text = format_instruction(item['inputs'], item['targets'],'')

                response = item['targets']
                #response = f'\n### Response: {response}'
                
                encodings = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=max_length)
                response_encodings = tokenizer(response, return_tensors='pt', truncation=True, max_length=max_length)

                response_len = response_encodings.input_ids.size(1)

                input_ids = encodings.input_ids.to(model.device)
                target_ids = input_ids.clone()
                target_ids[:, :-response_len] = -100

            if exp_type == 'domain':
                input_text = item['inputs']
                encodings = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=max_length)
                target_ids = encodings.input_ids.to(model.device)
                input_ids = target_ids.clone()
            
            outputs = model(input_ids, labels=target_ids)
            loss = outputs.loss

            
            #losses.append(loss)
            total_loss += loss.item()

    return torch.exp(torch.tensor(total_loss/ len(dataset))).item() #torch.exp(torch.stack(losses).mean())

## CLUSTERED RESULTS
#base_path = 'output/aya_dataset_400000_clustered_c20s2_i10_b16a1_l512_r8a16_20241002153817'
#
#paths = [base_path + '/checkpoint-1',
#         base_path + '/checkpoint-10',
#         base_path + '/checkpoint-50',
#         base_path + '/cluster_0_checkpoint-100',
#         base_path + '/cluster_1_checkpoint-100',
#         base_path + '/cluster_2_checkpoint-100',
#         base_path + '/cluster_3_checkpoint-100',
#         base_path + '/cluster_4_checkpoint-100',
#         base_path + '/cluster_0_checkpoint-150',
#         base_path + '/cluster_1_checkpoint-150',
#         base_path + '/cluster_2_checkpoint-150',
#         base_path + '/cluster_3_checkpoint-150',
#         base_path + '/cluster_4_checkpoint-150',
#         base_path + '/cluster_0_checkpoint-200',
#         base_path + '/cluster_1_checkpoint-200',
#         base_path + '/cluster_2_checkpoint-200',
#         base_path + '/cluster_3_checkpoint-200',
#         base_path + '/cluster_4_checkpoint-200']

###FEDAVG RESULTS
base_path = 'output/aya_dataset_400000_clustered_c20s2_i10_b16a1_l512_r8a16_20241003162141'
paths = [base_path + '/cluster_0_checkpoint-100',
         base_path + '/cluster_0_checkpoint-150',
         base_path + '/cluster_0_checkpoint-200']#

categories  = ['English', 'Swedish', 'German', 'Portuguese', 'Spanish']

results = []
df = pd.DataFrame(columns=['model', 'category', 'ppl'])

for category in categories:
    for path in paths:

        model, tokenizer = load_model(path, MODEL_NAME, DEVICE)
        test_dataset = load_eval_data(DATASET_NAME, EVALSET_LEN, category)

        perplexity = calculate_perplexity(model, tokenizer, test_dataset, max_length=512)

        model_eval = path.split('/')[-1]
        round = path.split('-')[-1]
        print(f'Perplexity {model_eval}: {perplexity}')

        results.append({'model': model_eval, 'round': round, 'category': category, 'ppl': perplexity})

    df = pd.DataFrame(results)
    df.to_csv('perplexity_federated.csv', index=False)


