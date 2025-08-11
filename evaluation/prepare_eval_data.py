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

sys.path.append(".")
from utils.template import TEMPLATE_DICT

def load_data(DATASET_NAME, tasks, eval=False):
    if DATASET_NAME == "databricks/databricks-dolly-15k":
        dataset = load_dataset(DATASET_NAME, split="train")
        dataset = dataset.train_test_split(test_size=0.2, seed=0)
        dataset = dataset['test'] if eval else dataset['train']
        dataset = dataset.filter(lambda x: x['category'] in tasks)
        dataset = dataset.map(dolly_format)
        return dataset

    if DATASET_NAME == "CohereForAI/aya_dataset":
 
        dataset = load_dataset(DATASET_NAME, split="train")
        languages = ['English', 'Dutch', 'Turkish', 'Portuguese', 'Spanish']
        dataset = dataset.filter(lambda x: x['language'] in languages)
        dataset = dataset.train_test_split(test_size=0.2, seed=0)
        dataset = dataset['test'] if eval else dataset['train']
        #tasks = [task.capitalize() for task in tasks]
        dataset = dataset.filter(lambda x: x['language'] == tasks)
        dataset = dataset.map(aya_format)
        return dataset

    if DATASET_NAME == 'multitask':
        if tasks == 'boolq' or 'boolq' in tasks:
            dataset = prepare_boolq(eval=eval).shuffle(seed=0)
            return dataset
        if tasks == 'webnlg' or 'webnlg' in tasks:
            dataset = prepare_webnlg(eval=eval).shuffle(seed=0)
            return dataset
        if tasks == 'samsum' or 'samsum' in tasks:
            dataset = prepare_samsum(eval=eval).shuffle(seed=0)
            return dataset
        if tasks == 'gigaword' or 'gigaword' in tasks:
            dataset = prepare_gigaword(eval=eval).shuffle(seed=0)
            return dataset
        if tasks == 'all_tasks' or 'all_tasks' in tasks:
            boolq = prepare_boolq(eval=eval).map(lambda x: {'instruction': x['instruction'], 'response': x['response'], 'task': 'boolq'})
            webnlg = prepare_webnlg(eval=eval).map(lambda x: {'instruction': x['instruction'], 'response': x['response'], 'task': 'webnlg'})
            samsum = prepare_samsum(eval=eval).map(lambda x: {'instruction': x['instruction'], 'response': x['response'], 'task': 'samsum'})
            gigaword = prepare_gigaword(eval=eval).map(lambda x: {'instruction': x['instruction'], 'response': x['response'], 'task': 'gigaword'})
            dataset = concatenate_datasets([boolq, webnlg, samsum, gigaword]).shuffle(seed=0)
            return dataset

def prepare_webnlg(eval=False):
    dataset = load_dataset('GEM/web_nlg', 'en', split='train')
    dataset = dataset.train_test_split(test_size=0.2, seed=0)
    dataset = dataset['test'] if eval else dataset['train']
    dataset = dataset.map(webnlg_format)
    return dataset

def prepare_boolq(eval=False):
    dataset = load_dataset('google/boolq', split='train')
    dataset = dataset.train_test_split(test_size=0.2, seed=0)
    dataset = dataset['test'] if eval else dataset['train']
    dataset = dataset.map(boolq_format)
    return dataset

def prepare_samsum(eval=False):
    dataset = load_dataset('Samsung/samsum', split='train', trust_remote_code=True)
    dataset = dataset.train_test_split(test_size=0.2, seed=0)
    dataset = dataset['test'] if eval else dataset['train']
    dataset = dataset.map(samsum_format)
    return dataset

def prepare_gigaword(eval=False):
    dataset = load_dataset('Harvard/gigaword', split='train', trust_remote_code=True)
    dataset = dataset.train_test_split(test_size=0.2, seed=0)
    dataset = dataset['test'] if eval else dataset['train']
    dataset = dataset.shuffle(seed=0)
    dataset = dataset.select(range(30000))
    dataset = dataset.map(gigaword_format)
    return dataset

def boolq_format(example):
    #example["instruction"] = example['passage'] + " Based on the passage, answer this question:" + example['question']
    example["instruction"] = example['passage'] + '-' + example['question']
    example["response"] = str(example['answer'])
    return example

def webnlg_format(example):
    example['input'] = str(example['input'])
    #example["instruction"] = "Organize this data into a readable text: " + example['input']
    example["instruction"] = example['input']
    example["response"] = example['target']
    return example

def samsum_format(example):
    #example["instruction"] = "Summarize this conversation: " + example['dialogue']
    example["instruction"] = example['dialogue']
    example["response"] = example['summary']
    return example

def gigaword_format(example):
    #example["instruction"] = "Summarize this text: " + example['document']
    example["instruction"] = example['document']
    example["response"] = example['summary']
    return example

def dolly_format(example):
    if example['context'] == "":
        example["inputs"] = example["instruction"]
    else:
        example["inputs"] = example["instruction"] + " " + example['context']
    return example

def aya_format(example):
    example["instruction"] = example['inputs']
    example["response"] = example['targets']
    return example


alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}"""

TEMPLATE_DICT = {
    'alpaca': (alpaca_template, '\n### Response:'),
}

def tokenize_function(examples):
    inputs = tokenizer(examples["inputs"], return_tensors="pt", padding='max_length', truncation=True, max_length=1024)
    targets = inputs.copy()
    return {
        "input_ids": inputs["input_ids"].squeeze(),
        "attention_mask": inputs["attention_mask"].squeeze(),
        "labels": targets["input_ids"].squeeze()
    }

def format_instruction(instruction, response, eos):
    template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}"""
    return template.format(instruction, response, eos)

def apply_template_to_dataset(dataset):
    dataset = dataset.map(lambda x: {'inputs': format_instruction(x, '', '')})
    return dataset


def get_formatting_prompts_func_test(template_name, eos_token):
    if template_name in TEMPLATE_DICT:
        overall_temp, response_temp = TEMPLATE_DICT[template_name]
        def formatting_prompts_func(example):
            text = overall_temp.format(example['instruction'], '', '')
            return text
    elif template_name == 'ag_news':
        formatting_prompts_func = None
        response_temp = None
    return formatting_prompts_func, response_temp