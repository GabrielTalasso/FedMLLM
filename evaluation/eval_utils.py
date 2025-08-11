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
def load_model(path, MODEL_NAME, DEVICE='cuda', adapter_name=None, global_dpa_path=None):
    # Configure 4-bit quantization using BitsAndBytes.
    bits_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    if adapter_name is not None:
        # Load base model in 4-bit mode.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bits_config,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(model, global_dpa_path, adapter_name='global')
        model = model.merge_and_unload()

        ckpts = glob.glob(path)
        latest_ckpt = max(ckpts, key=os.path.getctime)
        latest_ckpt = latest_ckpt + '/local'
        print(f"Loading model from {latest_ckpt}")
        model = PeftModel.from_pretrained(model, latest_ckpt, adapter_name=adapter_name)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    else:
        # Load model in 4-bit mode directly.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bits_config,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(model, path)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")

    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def evaluate_model(model, tokenizer, dataset, batch_size=8, device="cuda"):
    """
    Evaluates the model in one pass over the dataset, computing:
      - generated responses (via greedy argmax on logits),
      - average loss,
      - perplexity (exp(average loss)).
    
    Each example in the dataset should be a dict with keys:
      "inputs": the prompt,
      "targets": the expected output.
    
    This version uses a single forward pass (model(...)) for both computing the loss
    and generating the responses.
    """
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    total_examples = 0
    model_responses = []
    
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        
        # For each example, build the full text and record prompt lengths.
        full_texts = []
        prompt_lengths = []
        target_lengths = []
        for ex in batch:
            full_text = ex["inputs"] + ex["targets"]
            full_texts.append(full_text)
            # Compute length of prompt and target separately (without padding).
            prompt_ids = tokenizer(ex["inputs"], return_tensors="pt")["input_ids"][0]
            target_ids = tokenizer(ex["targets"], return_tensors="pt")["input_ids"][0]
            prompt_lengths.append(len(prompt_ids))
            target_lengths.append(len(target_ids))
        
        # Tokenize the concatenated texts.
        full_encodings = tokenizer(
            full_texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids_full = full_encodings["input_ids"].to(device)
        attention_mask_full = full_encodings["attention_mask"].to(device)
        
        # Create labels: mask out prompt tokens so that loss is computed only on target tokens.
        labels = input_ids_full.clone()
        for j, p_len in enumerate(prompt_lengths):
            if input_ids_full.shape[1] > p_len:
                labels[j, :p_len] = -100

        with torch.no_grad():
            # A single forward pass computes both logits and loss.
            outputs = model(input_ids_full, attention_mask=attention_mask_full, labels=labels)
            logits = outputs.logits  # shape: (batch_size, sequence_length, vocab_size)
        
        # The loss in outputs.loss is averaged over the non-masked tokens.
        # To accumulate a sum, we multiply by the number of examples in the batch.
        total_loss += outputs.loss.item() * len(batch)
        total_examples += len(batch)
        
        # For each example, extract predicted tokens for the target part by greedy argmax.
        for j, (p_len, t_len) in enumerate(zip(prompt_lengths, target_lengths)):
            # Get logits corresponding to token positions for the target.
            # Since the full input is [prompt + target], we take tokens at positions [p_len : p_len+t_len]
            target_logits = logits[j, p_len : p_len + t_len, :]
            pred_ids = target_logits.argmax(dim=-1)
            response = tokenizer.decode(pred_ids, skip_special_tokens=True)
            model_responses.append(response)
            
    avg_loss = total_loss / total_examples if total_examples > 0 else float("inf")
    perplexity = np.exp(avg_loss)
    return model_responses, avg_loss, perplexity