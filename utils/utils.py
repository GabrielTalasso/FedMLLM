import math
import sys
import os
from tqdm import tqdm
import numpy as np
import torch
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets
from accelerate import Accelerator
from torch.utils.data import DataLoader
import evaluate
import time
import glob

def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr

def apply_template_to_dataset(dataset, formatting_prompts_func):
    dataset = dataset.map(lambda x: {'inputs': formatting_prompts_func(x).split("### Response: ")[0] + "### Response: ", 'targets': x['response']})
    return dataset

def get_model_responses(model, tokenizer, dataset, batch_size=8):
    model_responses = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        #padding longest
        tokenized = tokenizer(batch['inputs'], padding_side='left', padding='longest',return_tensors='pt', truncation=True, max_length=1024)
        input_ids = tokenized['input_ids'].to('cuda')
        attention_mask = tokenized['attention_mask'].to('cuda')
        #print(input_ids[0], input_ids[1])
        with torch.no_grad():
            print(f"Generating responses for batch {i//batch_size + 1}/{len(dataset)//batch_size + 1}")
            outputs = model.generate(input_ids=input_ids, attention_mask = attention_mask, max_new_tokens=512, num_beams=1,
                                      do_sample=False, use_cache=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                      )
            batch_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            model_responses.extend(batch_responses)
    return model_responses

def calcule_rogue1(model_responses, dataset):
    metric = evaluate.load("rouge")
    references = [dataset[i]['targets'] for i in range(len(dataset))]
    predictions = [dataset[i]['model_responses'].split("### Response: ")[-1] for i in range(len(dataset))]

    scores = metric.compute(predictions=predictions, references=references)
    return scores

def default_evaluation(model, tokenizer, dataset, client_id, round, formatting_prompts_func, script_args, cluster_id=None):
    """
    Default evaluation function to compute model responses and ROUGE scores.
    """
    print("Evaluating model...")
    # Apply template to dataset
    dataset = apply_template_to_dataset(dataset, formatting_prompts_func)
    dataset_length = len(dataset)
    eval_responses = get_model_responses(model, tokenizer, dataset, batch_size=script_args.eval_batch_size)
    dataset_with_responses = dataset.select(range(len(dataset)))
    dataset_with_responses = dataset_with_responses.add_column('model_responses', eval_responses)
    scores = calcule_rogue1(eval_responses, dataset_with_responses)
    print(f"Evaluation scores: {scores}")

    #verify output directory
    if not os.path.exists(os.path.join(script_args.output_dir, "evals")):
        os.makedirs(os.path.join(script_args.output_dir, "evals"))
    # Save evaluation results
    with open(os.path.join(script_args.output_dir, f"evals/rouge_client_{client_id}_cluster_{cluster_id}_round_{round+1}.json"), 'w') as f:
        scores['dataset_length'] = dataset_length
        json.dump(scores, f, indent=4)


def save_dataset_test(dataset, script_args, client_id, round):
    """
    Save the test dataset for a specific client and round.
    """
    if not os.path.exists(os.path.join(script_args.output_dir, "clients_test_datasets")):
        os.makedirs(os.path.join(script_args.output_dir, "clients_test_datasets"))
    
    dataset.save_to_disk(os.path.join(script_args.output_dir, f"clients_test_datasets/client_{client_id}_round_{round}"))
