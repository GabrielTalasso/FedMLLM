"""
Clustered Federated Learning Simulation for Large Language Models.

This module orchestrates the complete clustered federated learning simulation
by setting up the experiment configuration, creating client and server components,
and running the Flower simulation.

The simulation follows the structure from main_sft_clustered.py but uses the
Flower federated learning framework for distributed training.
"""

import os
import gc
import copy
import torch
from typing import Dict, Any, Optional

from flwr.client import ClientApp
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context

# Import project modules
from utils import *
from utils.utils import default_evaluation, save_dataset_test, cosine_learning_rate
from federated_learning.split_dataset import split_dataset, get_dataset_this_round
from config import get_config, save_config, get_model_config, get_training_args

# Import our custom components
from client import create_client_fn
from server import create_server_strategy


def setup_experiment(config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

    script_args, fed_args, peft_config = get_config()

    # Update script_args settings with sensible defaults for testing
    if not config_overrides or 'max_steps' not in config_overrides:
        script_args.max_steps = 50  # Increased from 10 to 50 for better learning
    if not config_overrides or 'num_train_epochs' not in config_overrides:
        script_args.num_train_epochs = 1
    if not config_overrides or 'learning_rate' not in config_overrides:
        script_args.learning_rate = 1e-3  # Increased from 5e-4 to 1e-3
    if not config_overrides or 'batch_size' not in config_overrides:
        script_args.batch_size = 8  # Reduced from 16 to 8 for more gradient updates
    if not config_overrides or 'gradient_accumulation_steps' not in config_overrides:
        script_args.gradient_accumulation_steps = 2  # Increased from 1 to 2
    if not config_overrides or 'seq_length' not in config_overrides:
        script_args.seq_length = 1024
    if not config_overrides or 'dataset_name' not in config_overrides:
        script_args.dataset_name = "CohereForAI/aya_dataset"
    if not config_overrides or 'dataset_sample' not in config_overrides:
        script_args.dataset_sample = 400000
    if not config_overrides or 'output_dir' not in config_overrides:
        script_args.output_dir = "output"
    if not config_overrides or 'eval_batch_size' not in config_overrides:
        script_args.eval_batch_size = 128
    if not config_overrides or 'sim_alias' not in config_overrides:
        script_args.sim_alias = "testing_flower"
    if not config_overrides or 'peft_lora_r' not in config_overrides:
        script_args.peft_lora_r = 8
    if not config_overrides or 'peft_lora_alpha' not in config_overrides:
        script_args.peft_lora_alpha = 16
    if not config_overrides or 'train_split' not in config_overrides:
        script_args.train_split = 0.8
    if not config_overrides or 'model_name_or_path' not in config_overrides:
        script_args.model_name_or_path = 'HuggingFaceTB/SmolLM-135M'

    # Update fed_args with sensible defaults for testing
    if not config_overrides or 'fed_alg' not in config_overrides:
        fed_args.fed_alg = "clustered"
    if not config_overrides or 'num_rounds' not in config_overrides:
        fed_args.num_rounds = 10
    if not config_overrides or 'num_clients' not in config_overrides:
        fed_args.num_clients = 5
    if not config_overrides or 'sample_clients' not in config_overrides:
        fed_args.sample_clients = 5
    if not config_overrides or 'split_strategy' not in config_overrides:
        fed_args.split_strategy = "language_clusters"
    if not config_overrides or 'sim_round' not in config_overrides:
        fed_args.sim_round = 3  # Changed from 1 to 3 to allow more pre-clustering learning
    if not config_overrides or 'n_clusters' not in config_overrides:
        fed_args.n_clusters = 5
    if not config_overrides or 'evaluation_rounds' not in config_overrides:
        fed_args.evaluation_rounds = "10,25,50,75,100"
    if not config_overrides or 'evaluation_mode' not in config_overrides:
        fed_args.evaluation_mode = "local"

    training_args = get_training_args(script_args, script_args.learning_rate)
    save_config(script_args, fed_args)
    
    print(f"Final script_args: {script_args}")
    print(f"Final fed_args: {fed_args}")
    
    dataset, dataset_test = get_dataset(script_args.dataset_name, script_args.local_data_dir, script_args.train_split)
    dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)
    dataset_test = process_sft_dataset(script_args.dataset_name, dataset_test, script_args.dataset_sample)

    local_datasets = split_dataset(fed_args, script_args, dataset)
    
    if fed_args.evaluation_mode == "local":
        local_datasets_test = split_dataset(fed_args, script_args, dataset_test)
    elif fed_args.evaluation_mode == "global":
        aux_split_strategy = fed_args.split_strategy
        fed_args.split_strategy = fed_args.split_strategy.split('_')[0] + '_iid'
        local_datasets_test = split_dataset(fed_args, script_args, dataset_test)
        fed_args.split_strategy = aux_split_strategy
    
    print(f"Created {len(local_datasets)} local datasets for clients")
    for i, ds in enumerate(local_datasets):
        print(f"  Client {i}: {len(ds)} training samples")
    
    # Clean up memory
    del dataset, dataset_test
    gc.collect()
    
    device_map, quantization_config, torch_dtype = get_model_config(script_args)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, prepare_model_for_kbit_training
    from trl import DataCollatorForCompletionOnlyLM
    
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )
    
    if script_args.load_in_8bit or script_args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.config.use_cache = False
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.eos_token == tokenizer.unk_token or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    
    model.resize_token_embeddings(len(tokenizer))
    
    formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
    if response_template:
        response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
        data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
        packing = False
    else:
        data_collator = None
        packing = True
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(script_args.output_dir, exist_ok=True)
    
    experiment_config = {
        'script_args': script_args,
        'fed_args': fed_args,
        'training_args': training_args,
        'model': model,
        'tokenizer': tokenizer,
        'local_datasets': local_datasets,
        'local_datasets_test': local_datasets_test,
        'formatting_prompts_func': formatting_prompts_func,
        'data_collator': data_collator,
        'packing': packing,
        'peft_config': peft_config,
        'device': device
    }
    
    print("Experiment setup completed successfully!")
    return experiment_config


def create_simulation_components(experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    
    # Create client factory function
    client_fn = create_client_fn(experiment_config)
    
    # Create server strategy
    strategy = create_server_strategy(experiment_config)
    
    # Configure simulation parameters
    client_resources = {
        "num_cpus": experiment_config.get('client_resources_cpus', 1), 
        "num_gpus": experiment_config.get('client_resources_gpus', 1)
    }
    
    # Create client app
    client_app = ClientApp(client_fn=client_fn)
    
    # Create server app
    def server_fn(context: Context):
        config = ServerConfig(num_rounds=experiment_config['fed_args'].num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)
    
    server_app = ServerApp(server_fn=server_fn)
    
    # Save initial model (mimics the checkpoint-0 save in main_sft_clustered.py)
    initial_model_path = os.path.join(experiment_config['script_args'].output_dir, "checkpoint-0")
    os.makedirs(initial_model_path, exist_ok=True)
    experiment_config['model'].save_pretrained(initial_model_path)
    experiment_config['tokenizer'].save_pretrained(initial_model_path)
    
    print("Simulation components created successfully!")
    
    return {
        'client_app': client_app,
        'server_app': server_app,
        'client_resources': client_resources,
        'strategy': strategy,
        'client_fn': client_fn,
        'experiment_config': experiment_config
    }


def run_simulation(num_rounds: int = 10, config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
 
    print(f"\n=== Starting Clustered FedMLLM Simulation Setup ({num_rounds} rounds) ===")
    
    # Setup experiment configuration
    # This replaces all the initialization code in main_sft_clustered.py
    if config_overrides is None:
        config_overrides = {}
    
    # Override num_rounds in fed_args
    config_overrides['num_rounds'] = num_rounds
    
    experiment_config = setup_experiment(config_overrides)
    
    # Create simulation components
    simulation_components = create_simulation_components(experiment_config)
    
    print(f"\nSimulation Configuration:")
    print(f"  - Clients: {experiment_config['fed_args'].num_clients}")
    print(f"  - Clustering round: {experiment_config['fed_args'].sim_round}")
    print(f"  - Number of clusters: {experiment_config['fed_args'].n_clusters}")
    print(f"  - Rounds: {num_rounds}")
    print(f"  - Model: {experiment_config['script_args'].model_name_or_path}")
    print(f"  - Dataset: {experiment_config['script_args'].dataset_name}")
    print(f"  - Output dir: {experiment_config['script_args'].output_dir}")
    print(f"  - Split strategy: {experiment_config['fed_args'].split_strategy}")
    print(f"  - Device: {experiment_config['device']}")
    
    print("\nSimulation setup complete. Ready to run!")
    print("\nTo run the actual simulation, call:")
    print("  run_flower_simulation_actual(simulation_components)")
    
    return simulation_components


def run_flower_simulation_actual(simulation_components: Dict[str, Any]) -> Optional[Any]:
   
    try:
        from flwr.simulation import run_simulation as flwr_run_simulation
        
        # Extract components
        client_app = simulation_components['client_app']
        server_app = simulation_components['server_app']
        client_resources = simulation_components['client_resources']
        experiment_config = simulation_components['experiment_config']
        
        # Run the simulation
        history = flwr_run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=experiment_config['fed_args'].num_clients,
             backend_config={"client_resources": {"num_cpus": 32, "num_gpus": 1}}
        )
        
        print("\n=== Simulation Completed Successfully ===")
        print(f"History: {history}")
        
        return history
        
    except ImportError:
        print("\nFlower simulation not available. Install with: pip install flwr[simulation]")
        return None
    except Exception as e:
        print(f"\nSimulation failed: {e}")
        print("This might be due to missing dependencies or configuration issues.")
        print("You can still use the setup components for testing individual parts.")
        return None


def main():
   
    import argparse
    
    parser = argparse.ArgumentParser(description="Clustered Federated Learning for Multilingual LLMs")
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of federated learning rounds")
    parser.add_argument("--num_clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--sim_round", type=int, default=1, help="Round at which clustering happens")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--sample_clients", type=int, default=5, help="Number of clients to sample per round")
    parser.add_argument("--split_strategy", type=str, default="language_clusters", help="Data split strategy")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--test_only", action="store_true", help="Only run tests, don't start simulation")
    parser.add_argument("--run_simulation", action="store_true", help="Run the actual simulation")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M", help="Model name or path")
    parser.add_argument("--model_name_or_path", type=str, default="HuggingFaceTB/SmolLM-135M", help="Model name or path")
    parser.add_argument("--use_peft", type=str, default="True", help="Whether to use PEFT")
    parser.add_argument("--peft_lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--peft_lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--load_in_4bit", type=str, default="True", help="Load model in 4-bit")
    parser.add_argument("--load_in_8bit", type=str, default="False", help="Load model in 8-bit")
    
    # Dataset configuration
    parser.add_argument("--dataset_name", type=str, default="CohereForAI/aya_dataset", help="Dataset name")
    parser.add_argument("--dataset_sample", type=int, default=400000, help="Dataset sample size")
    parser.add_argument("--train_split", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--template", type=str, default="alpaca", help="Prompt template")
    
    # Training configuration
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum training steps")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--seq_length", type=int, default=1024, help="Sequence length")
    
    # Federated learning configuration
    parser.add_argument("--fed_alg", type=str, default="clustered", help="Federated learning algorithm")
    
    # Evaluation configuration
    parser.add_argument("--evaluation_rounds", type=str, default="10,25,50,75,100", help="Evaluation rounds")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Evaluation batch size")
    parser.add_argument("--evaluation_mode", type=str, default="local", help="Evaluation mode")
    
    # Simulation configuration
    parser.add_argument("--sim_alias", type=str, default="flower_clustered", help="Simulation alias")
    parser.add_argument("--client_resources_cpus", type=int, default=1, help="CPU resources per client")
    parser.add_argument("--client_resources_gpus", type=float, default=0.1, help="GPU resources per client")
    
    args = parser.parse_args()
    
    # Helper function to convert string booleans
    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            return v
    
    # Prepare configuration overrides
    config_overrides = {
        'num_clients': args.num_clients,
        'sample_clients': args.sample_clients,
        'sim_round': args.sim_round,
        'n_clusters': args.n_clusters,
        'model_name_or_path': args.model_name_or_path or args.model_name,
        'dataset_name': args.dataset_name,
        'dataset_sample': args.dataset_sample,
        'split_strategy': args.split_strategy,
        'output_dir': args.output_dir,
        'learning_rate': args.learning_rate,
        'max_steps': args.max_steps,
        'num_train_epochs': args.num_train_epochs,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'seq_length': args.seq_length,
        'peft_lora_r': args.peft_lora_r,
        'peft_lora_alpha': args.peft_lora_alpha,
        'train_split': args.train_split,
        'template': args.template,
        'fed_alg': args.fed_alg,
        'evaluation_rounds': args.evaluation_rounds,
        'eval_batch_size': args.eval_batch_size,
        'evaluation_mode': args.evaluation_mode,
        'sim_alias': args.sim_alias,
        'load_in_4bit': str_to_bool(args.load_in_4bit),
        'load_in_8bit': str_to_bool(args.load_in_8bit),
        'use_peft': str_to_bool(args.use_peft),
        'client_resources_cpus': args.client_resources_cpus,
        'client_resources_gpus': args.client_resources_gpus,
    }
    
    # Run simulation setup
    simulation_components = run_simulation(args.num_rounds, config_overrides)
    
    if args.run_simulation:
        # Run actual simulation
        history = run_flower_simulation_actual(simulation_components)
        if history:
            print(f"\nSimulation completed. Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
