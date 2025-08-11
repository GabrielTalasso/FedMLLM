import os
import sys
import json
import glob
import numpy as np
import torch
import evaluate
from tqdm import tqdm

sys.path.append(".")
from evaluation.eval_utils import load_model, evaluate_model
from evaluation.prepare_eval_data import load_data, get_formatting_prompts_func_test

def unified_model_evaluation(
    model_path,
    base_model,
    dataset_name,
    task,
    device="cuda",
    eval_len=100,
    eval_rouge=True, 
    eval_perplexity=True,
    output_dir=None,
    is_dpa=False,
    global_dpa_path=None,
    batch_size=8
):
    """
    Evaluate a model using functions from eval_utils.py and prepare_eval_data.py.
    
    Args:
        model_path: Path to the model checkpoint
        base_model: Base model name
        dataset_name: Name of the dataset to evaluate on
        task: Task name or language for filtering the dataset
        device: Device to run evaluation on
        eval_len: Number of examples to evaluate
        eval_rouge: Whether to compute ROUGE scores
        eval_perplexity: Whether to compute perplexity
        output_dir: Directory to save evaluation results
        is_dpa: Whether the model uses domain-personalized adapters
        global_dpa_path: Path to global adapter (for DPA models)
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Evaluating model: {model_path} on task: {task}")
    print(f"Base model: {base_model}")
    print(f"Dataset: {dataset_name}")
    
    # Load model using eval_utils.load_model
    if is_dpa:
        model, tokenizer = load_model(
            model_path, 
            base_model, 
            DEVICE=device, 
            adapter_name='local', 
            global_dpa_path=global_dpa_path
        )
    else:
        model, tokenizer = load_model(
            model_path, 
            base_model, 
            DEVICE=device
        )
    
    # Load and format dataset using prepare_eval_data.load_data
    dataset = load_data(dataset_name, task, eval=True)
    dataset = dataset.select(range(eval_len))
    
    # Apply formatting template using prepare_eval_data
    formatting_prompts_func, _ = get_formatting_prompts_func_test('alpaca', '\n### Response:')
    formatted_dataset = dataset.map(
        lambda x: {
            'inputs': formatting_prompts_func(x),
            'targets': x['response']
        }
    )
    
    # Create evaluation dataset with the format expected by evaluate_model
    eval_dataset = [
        {"inputs": item["inputs"], "targets": item["targets"]} 
        for item in formatted_dataset
    ]
    
    # Create output directory
    model_name = os.path.basename(model_path) if os.path.isdir(model_path) else os.path.basename(os.path.dirname(model_path))
    base_model_name = base_model.split('/')[-1]
    
    if output_dir is None:
        output_dir = model_path
        
    if is_dpa:
        ckpts = glob.glob(output_dir)
        latest_ckpt = max(ckpts, key=os.path.getctime)
        latest_ckpt = latest_ckpt + '/local'
        out_dir = os.path.join(latest_ckpt, f"{model_name}_{base_model_name}_{task}")
    else:
        out_dir = os.path.join(output_dir, f"{model_name}_{base_model_name}_{task}")
    
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    
    # Evaluate using eval_utils.evaluate_model
    model_responses, avg_loss, perplexity = evaluate_model(
        model, tokenizer, eval_dataset, batch_size=batch_size, device=device
    )
    
    # Save model responses
    dataset_with_responses = formatted_dataset.add_column('model_responses', model_responses)
    dataset_with_responses.save_to_disk(os.path.join(out_dir, "responses"))
    
    if eval_rouge:
        print("Calculating ROUGE scores...")
        metric = evaluate.load("rouge")
        references = [item['targets'] for item in formatted_dataset]
        predictions = model_responses
        scores = metric.compute(predictions=predictions, references=references)
        print(f"ROUGE scores: {scores}")
        
        with open(os.path.join(out_dir, "rouge.json"), 'w') as f:
            json.dump(scores, f)
        results['rouge'] = scores
    
    if eval_perplexity:
        print("Calculating perplexity...")
        # The perplexity is already computed by evaluate_model
        perplexity_results = {
            "average_perplexity": float(perplexity),
            "loss": float(avg_loss)
        }
        
        with open(os.path.join(out_dir, "perplexity.json"), 'w') as f:
            json.dump(perplexity_results, f)
        results['perplexity'] = perplexity_results
    
    # Save combined results
    with open(os.path.join(out_dir, "results.json"), 'w') as f:
        json.dump(results, f)
    
    print(f"Evaluation complete. Results saved to {out_dir}")
    return results


def evaluate_multiple_models(
    model_list,
    base_model,
    dataset_name,
    task_list,
    **kwargs
):
    """
    Evaluate multiple models on multiple tasks.
    
    Args:
        model_list: List of model paths to evaluate
        base_model: Base model name
        dataset_name: Name of the dataset to evaluate on
        task_list: List of tasks or languages to evaluate on
        **kwargs: Additional arguments to pass to unified_model_evaluation
        
    Returns:
        Dictionary mapping model paths to evaluation results
    """
    all_results = {}
    
    for model_path in model_list:
        performance_for_model = {}
        for task in task_list:
            print(f"\nEvaluating Model: {model_path} for Task: {task}")
            results = unified_model_evaluation(
                model_path=model_path,
                base_model=base_model,
                dataset_name=dataset_name,
                task=task,
                **kwargs
            )
            performance_for_model[task] = results
        
        # Save performance summary
        is_dpa = kwargs.get('is_dpa', False)
        if is_dpa:
            ckpts = glob.glob(model_path)
            latest_ckpt = max(ckpts, key=os.path.getctime)
            latest_ckpt = latest_ckpt + '/local'
            performance_json_path = os.path.join(latest_ckpt, "performance.json")
        else:
            performance_json_path = os.path.join(model_path, "performance.json")
        
        with open(performance_json_path, "w") as f:
            json.dump(performance_for_model, f)
        
        print(f"Saved performance results for model {model_path} to {performance_json_path}")
        all_results[model_path] = performance_for_model
    
    return all_results


if __name__ == "__main__":
    # Example usage
    is_dpa = False
    global_dpa_path = None

    # Define models to evaluate
    path_model_clustered = 'output_aya/Llama-3.2-1B/clustered_in_round_1_aya_dataset_clustered_c20s5_i10_b16a1_l1024_r8a16'
    model_list = [
        path_model_clustered + '/cluster_0_checkpoint-200',
        path_model_clustered + '/cluster_1_checkpoint-200',
        path_model_clustered + '/cluster_2_checkpoint-200',
        path_model_clustered + '/cluster_3_checkpoint-200',
        path_model_clustered + '/cluster_4_checkpoint-200',
    ]

    base_model = 'unsloth/Llama-3.2-1B'
    dataset_name = 'CohereForAI/aya_dataset'
    dataset_name = 'multitask'
    task_list = ['English', 'Dutch', 'Turkish', 'Portuguese', 'Spanish']
    task_list = []
    
    # Run evaluation
    evaluate_multiple_models(
        model_list=model_list,
        base_model=base_model,
        dataset_name=dataset_name,
        task_list=task_list,
        device='cuda',
        eval_len=100,
        eval_rouge=True,
        eval_perplexity=True,
        batch_size=8,
        is_dpa=is_dpa,
        global_dpa_path=global_dpa_path
    )