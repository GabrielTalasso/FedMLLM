"""
Clustered Federated Learning Client for Large Language Models.

This module implements the client-side logic for clustered federated learning
following the approach from main_sft_clustered.py but adapted for Flower framework.

The client handles:
1. Pre-clustering: Normal federated learning with global model
2. Clustering phase: Client adapters are analyzed for similarity  
3. Post-clustering: Client receives cluster-specific model
"""

import os
import copy
import torch
import numpy as np
from typing import List, Dict, Any, Tuple

from flwr.client import NumPyClient
from flwr.common import Parameters, parameters_to_ndarrays
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from datasets import Dataset
from transformers import AutoTokenizer

# Import project utilities
from utils.utils import cosine_learning_rate, default_evaluation, save_dataset_test
from federated_learning.split_dataset import get_dataset_this_round
from federated_learning.fed_local_sft import get_fed_local_sft_trainer


class ClusteredFedMLLMClient(NumPyClient):
    
    def __init__(self, 
                 cid: int, 
                 model: torch.nn.Module,
                 tokenizer: AutoTokenizer,
                 local_dataset: Dataset,
                 local_dataset_test: Dataset,
                 script_args,
                 fed_args,
                 training_args,
                 formatting_prompts_func,
                 data_collator=None,
                 packing: bool = True,
                 device: torch.device = None,
                 output_dir: str = "./output"):
        
        self.cid = cid
        self.model = model
        self.tokenizer = tokenizer
        self.local_dataset = local_dataset
        self.local_dataset_test = local_dataset_test
        self.script_args = script_args
        self.fed_args = fed_args
        self.training_args = training_args
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.packing = packing
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        
        # Cluster information - will be set by server
        self.cluster_id = ''
        self.training_losses = []
        
        print(f"Client {self.cid} initialized with {len(self.local_dataset)} training samples")
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Extract model parameters as numpy arrays (LoRA adapter parameters only)."""
        state_dict = get_peft_model_state_dict(self.model)
        # Ensure parameters are on CPU before converting to numpy
        cpu_params = []
        for val in state_dict.values():
            if hasattr(val, 'cpu'):
                cpu_params.append(val.cpu().detach().numpy())
            else:
                cpu_params.append(np.array(val))
        return cpu_params
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays (LoRA adapter parameters only)."""
        # Get current state dict to preserve structure and device
        current_state_dict = get_peft_model_state_dict(self.model)
        
        if len(parameters) != len(current_state_dict):
            print(f"Warning: Parameter count mismatch. Expected {len(current_state_dict)}, got {len(parameters)}")
            return
        
        # Create new state dict with proper device placement
        new_state_dict = {}
        param_keys = list(current_state_dict.keys())
        
        for i, (key, param_array) in enumerate(zip(param_keys, parameters)):
            # Convert numpy to tensor and move to same device as current parameter
            current_param = current_state_dict[key]
            new_tensor = torch.from_numpy(param_array).to(device=current_param.device, dtype=current_param.dtype)
            new_state_dict[key] = new_tensor
        
        # Apply the new state dict
        set_peft_model_state_dict(self.model, new_state_dict)
    
    def fit(self, parameters: Parameters, config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        
        current_round = config["current_round"]
        total_rounds = config["total_rounds"]
        is_clustering_round = config.get("is_clustering_round", False)
        cluster_id = config.get("cluster_id", None)
        
        print(f"\n=== Client {self.cid} - Round {current_round} ===")
        
        # Update cluster information if provided (post-clustering phase)
        if cluster_id is not None:
            self.cluster_id = cluster_id
            print(f"Client {self.cid} assigned to cluster {cluster_id}")
        
        # Set model parameters (global or cluster-specific)
        if parameters is not None:
            print(f"Client {self.cid}: Received {len(parameters)} parameters")
            self.set_parameters(parameters)
            
            # Verify parameters were set correctly
            current_params = self.get_parameters(config)
            param_norms = [np.linalg.norm(p) for p in current_params]
            print(f"Client {self.cid}: Parameter norms after setting: {param_norms[:3]}...")  # Show first 3 norms
        else:
            print(f"Client {self.cid}: No parameters received!")
            
        self.model.to(self.device)
        
        # Get subset of data for this round
        sub_dataset = get_dataset_this_round(
            self.local_dataset, current_round, self.fed_args, self.script_args
        )
        
        # Update learning rate with cosine schedule  
        new_lr = cosine_learning_rate(
            current_round, total_rounds, self.script_args.learning_rate, 1e-5
        )
        
        # Update training arguments
        updated_training_args = copy.deepcopy(self.training_args)
        updated_training_args.learning_rate = new_lr
        updated_training_args.output_dir = os.path.join(
            self.script_args.output_dir, f"client_{self.cid}_round_{current_round}"
        )
        
        # Create federated SFT trainer
        # This mimics the trainer creation in main_sft_clustered.py
        trainer = get_fed_local_sft_trainer(
            script_args=self.script_args,
            fed_args=self.fed_args,
            model=self.model,
            tokenizer=self.tokenizer,
            training_args=updated_training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
            global_dict=None,  # Not needed for clustering approach
            local_auxiliary=None,  # Not using SCAFFOLD
            global_auxiliary=None,
            packing=self.packing,
        )
        
        print(f"Training client {self.cid} with lr={new_lr:.6f}")
        
        # Train the model
        results = trainer.train()
        self.training_losses.append(results.training_loss)
        
        # Save adapter for clustering analysis if this is the clustering round
        if is_clustering_round:
            self._save_adapter_for_clustering(current_round, trainer)
        
        # Extract updated parameters (LoRA adapter state dict)
        updated_parameters = self.get_parameters(config)
        
        # Verify we actually have parameters to return
        if not updated_parameters:
            print(f"Warning: Client {self.cid} has no parameters to return!")
        else:
            # Print parameter magnitudes for debugging
            param_norms = [np.linalg.norm(p) for p in updated_parameters]
            print(f"Client {self.cid}: Returning {len(updated_parameters)} parameters with norms: {param_norms[:3]}...")
        
        dataset_size = len(sub_dataset)
        
        # Prepare metrics
        metrics = {
            "training_loss": results.training_loss,
            "dataset_size": dataset_size,
            "cluster_id": self.cluster_id,
            "learning_rate": new_lr
        }
        
        print(f"Client {self.cid} completed training. Loss: {results.training_loss:.4f}")
        #print(len(updated_parameters))
        return updated_parameters, dataset_size, metrics
    
    def evaluate(self, parameters: Parameters, config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        current_round = config["current_round"]
        
        print(f"Client {self.cid}: Starting evaluation for round {current_round}")
        
        # Set model parameters
        if parameters is not None:
            self.set_parameters(parameters)
            print(f"Client {self.cid}: Applied parameters for evaluation")
        else:
            print(f"Client {self.cid}: Warning - No parameters received for evaluation!")
            
        self.model.to(self.device)
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Prepare test dataset
        # This mimics the test dataset preparation in main_sft_clustered.py
        sub_dataset_test = self.local_dataset_test.shuffle(seed=current_round)
        max_eval_size = getattr(self.script_args, 'max_eval_size', 100)
        if max_eval_size < len(sub_dataset_test):
            sub_dataset_test = sub_dataset_test.select(range(max_eval_size))
        
        print(f"Evaluating client {self.cid} on {len(sub_dataset_test)} samples")
        
        # Save test dataset (mimics save_dataset_test call)
        if (current_round + 1) in [int(x) for x in self.fed_args.evaluation_rounds.split(",")]:
            save_dataset_test(sub_dataset_test, self.script_args, self.cid, current_round)
        
            # Perform evaluation using the default evaluation function
            eval_results = default_evaluation(
                model=self.model,
                tokenizer=self.tokenizer,
                dataset=sub_dataset_test,
                client_id=self.cid,
                round=current_round,
                formatting_prompts_func=self.formatting_prompts_func,
                script_args=self.script_args,
                cluster_id=self.cluster_id
            )
        
            # Extract loss (for simplicity, we'll use a dummy loss since default_evaluation returns ROUGE scores)
            loss = eval_results.get('eval_loss', 0.0) if isinstance(eval_results, dict) else 0.0
        
        else:
            loss = 0.0

        # Return model to training mode
        self.model.train()

        metrics = {
            "eval_loss": loss,
            "cluster_id": self.cluster_id,
            "eval_samples": len(sub_dataset_test)
        }
        
        return loss, len(sub_dataset_test), metrics
    
    def _save_adapter_for_clustering(self, round_idx: int, trainer) -> None:

        output_dir = os.path.join(self.output_dir, "clients_adapters")
        os.makedirs(output_dir, exist_ok=True)
        
        adapter_path = os.path.join(output_dir, f"checkpoint-{round_idx}_client{self.cid}")
        
        # Save the adapter using trainer's save_model method
        trainer.save_model(adapter_path)
        
        print(f"Saved adapter for client {self.cid} at round {round_idx}")


def create_client_fn(experiment_config):
    """
    Create a client function for the Flower simulation.
    
    This function creates a closure that holds the experiment configuration
    and returns a client factory function for Flower.
    
    Args:
        experiment_config: Dictionary containing all experiment configuration:
            - model: Base model
            - tokenizer: Tokenizer
            - local_datasets: List of local datasets for each client
            - local_datasets_test: List of local test datasets for each client
            - script_args: Script arguments
            - fed_args: Federated learning arguments
            - training_args: Training arguments
            - formatting_prompts_func: Function to format prompts
            - data_collator: Data collator
            - packing: Whether to use packing
            
    Returns:
        Client factory function for Flower
    """
    from flwr.common import Context
    
    def client_fn(context: Context) -> ClusteredFedMLLMClient:
        """Create a client instance with the specified configuration."""
        cid = int(context.node_config["partition-id"])
        
        # Create a copy of the model for this client to avoid state sharing
        model_copy = copy.deepcopy(experiment_config['model'])
        
        # Get client's local dataset
        local_dataset = experiment_config['local_datasets'][cid]
        local_dataset_test = experiment_config['local_datasets_test'][cid]
        
        # Create client instance following the ClusteredFedMLLMClient pattern
        client = ClusteredFedMLLMClient(
            cid=cid,
            model=model_copy,
            tokenizer=experiment_config['tokenizer'],
            local_dataset=local_dataset,
            local_dataset_test=local_dataset_test,
            script_args=experiment_config['script_args'],
            fed_args=experiment_config['fed_args'],
            training_args=copy.deepcopy(experiment_config['training_args']),
            formatting_prompts_func=experiment_config['formatting_prompts_func'],
            data_collator=experiment_config['data_collator'],
            packing=experiment_config['packing'],
            device=experiment_config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            output_dir=experiment_config['script_args'].output_dir
        )
        
        return client
    
    return client_fn
