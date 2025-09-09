"""
Clustered Federated Learning Server Strategy for Large Language Models.

This module implements the server-side strategy for clustered federated learning
following the approach from main_sft_clustered.py and federated_learning/fed_global.py
but adapted for Flower framework.

The strategy handles:
1. Pre-clustering phase: Standard FedAvg aggregation (round < sim_round)
2. Clustering phase: Analyze client similarities and create clusters (round == sim_round)  
3. Post-clustering phase: Maintain cluster-specific models (round > sim_round)
"""

import os
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional

from flwr.server.strategy import Strategy, FedAvg
from flwr.common import (
    Parameters, 
    FitRes, 
    FitIns, 
    EvaluateRes,
    EvaluateIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)

# Import project utilities for clustering
from federated_learning.fed_clustered import calculate_similarity, make_clusters
from peft import get_peft_model_state_dict
from safetensors.torch import save_file


class ClusteredFedMLLMStrategy(FedAvg):
    """
    Clustered Federated Learning Strategy for Large Language Models.
    
    This strategy implements the three-phase approach from main_sft_clustered.py:
    1. Pre-clustering phase: Standard FedAvg aggregation (round < sim_round)
    2. Clustering phase: Analyze client similarities and create clusters (round == sim_round)  
    3. Post-clustering phase: Maintain cluster-specific models (round > sim_round)
    
    The aggregation logic is adapted from federated_learning/fed_global.py
    """
    
    def __init__(self, 
                 initial_parameters: Parameters,
                 peft_state_dict,
                 num_clients: int,
                 sim_round: int,
                 n_clusters: int,
                 output_dir: str,
                 fraction_fit: float = 1.0,
                 fraction_evaluate: float = 1.0,
                 min_fit_clients: int = 2,
                 min_evaluate_clients: int = 2,
                 min_available_clients: int = 2,
                 script_args=None,
                 fed_args=None):

        self.initial_parameters = initial_parameters
        self.num_clients = num_clients
        self.sim_round = sim_round
        self.n_clusters = n_clusters
        self.output_dir = output_dir
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.script_args = script_args
        self.fed_args = fed_args
        
        self.current_round = 0
        self.client_clusters = {}  
        self.cluster_models = {}  
        self.global_model = initial_parameters 
        self.cluster_assignments = None  
        self.peft_state_dict = peft_state_dict
        
        self.training_metrics = []
        
        # Save initial global model
        self._save_initial_global_model()
        
    
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return self.initial_parameters
    
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[Any, FitIns]]:

        self.current_round = server_round
        
        if server_round == self.sim_round:
            # For clustering round, include all clients
            sample_size = self.num_clients
        else:
            sample_size = max(int(self.num_clients * self.fraction_fit), self.min_fit_clients)
        
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_available_clients)
        
        # Prepare configuration for clients
        config = {
            "current_round": server_round,
            "total_rounds": self.fed_args.num_rounds if self.fed_args else 10,
            "is_clustering_round": server_round == self.sim_round,
        }
        
        # Determine which parameters to send to each client
        fit_configurations = []
        
        if server_round < self.sim_round:
            print(f"Round {server_round}: Pre-clustering phase - sending global model to all clients")
            for client in clients:
                fit_configurations.append((client, FitIns(parameters, config)))
                
        elif server_round == self.sim_round:
            print(f"Round {server_round}: Clustering phase - sending global model to all clients")
            for client in clients:
                fit_configurations.append((client, FitIns(parameters, config)))
                
        else:
            print(f"Round {server_round}: Post-clustering phase - sending cluster-specific models")
            for client in clients:
                client_id = int(client.cid)
                cluster_id = self.client_clusters.get(client_id, 0)
                cluster_params = self.cluster_models.get(cluster_id, self.global_model)
                
                client_config = config.copy()
                client_config["cluster_id"] = cluster_id
                
                fit_configurations.append((client, FitIns(cluster_params, client_config)))
        
        return fit_configurations
    
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[Any, EvaluateIns]]:
  
        # Sample clients for evaluation
        sample_size = max(int(self.num_clients * self.fraction_evaluate), self.min_evaluate_clients)
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_available_clients)
        
        config = {"current_round": server_round}
        
        # Send appropriate model to each client for evaluation
        evaluate_configurations = []
        
        if server_round <= self.sim_round:
            # Send global model for evaluation
            for client in clients:
                evaluate_configurations.append((client, EvaluateIns(self.global_model, config)))
        else:
            # Send cluster-specific models for evaluation
            for client in clients:
                client_id = int(client.cid)
                cluster_id = self.client_clusters.get(client_id, 0)
                cluster_params = self.cluster_models.get(cluster_id, self.global_model)
                evaluate_configurations.append((client, EvaluateIns(cluster_params, config)))
        
        return evaluate_configurations
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[Any, FitRes]], failures: List[Any]) -> Tuple[Optional[Parameters], Dict[str, Any]]:

        if not results:
            return None, {}
        
        if server_round < self.sim_round:
            return self._aggregate_fedavg(results)
            
        elif server_round == self.sim_round:
            return self._aggregate_with_clustering(results)
            
        else:
            return self._aggregate_clustered(results)

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[Any, EvaluateRes]], failures: List[Any]) -> Tuple[Optional[float], Dict[str, Any]]:

        if not results:
            return None, {}
        
        # Calculate average loss
        total_loss = sum([res.loss * res.num_examples for _, res in results])
        total_examples = sum([res.num_examples for _, res in results])
        avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
        
        # Aggregate metrics by cluster if in post-clustering phase
        metrics = {"avg_loss": avg_loss, "total_examples": total_examples}
        
        if server_round > self.sim_round:
            cluster_metrics = {}
            for client, res in results:
                client_id = int(client.cid)
                cluster_id = self.client_clusters.get(client_id, 0)
                
                if cluster_id not in cluster_metrics:
                    cluster_metrics[cluster_id] = {"loss": 0.0, "examples": 0, "clients": 0}
                
                cluster_metrics[cluster_id]["loss"] += res.loss * res.num_examples
                cluster_metrics[cluster_id]["examples"] += res.num_examples
                cluster_metrics[cluster_id]["clients"] += 1
            
            # Calculate average loss per cluster
            for cluster_id in cluster_metrics:
                cluster_metrics[cluster_id]["avg_loss"] = (
                    cluster_metrics[cluster_id]["loss"] / cluster_metrics[cluster_id]["examples"]
                    if cluster_metrics[cluster_id]["examples"] > 0 else 0.0
                )
            
            metrics["cluster_metrics"] = cluster_metrics
        
        print(f"Round {server_round} evaluation - Average loss: {avg_loss:.4f}")
        
        return avg_loss, metrics
    
    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Any]]]:
        #without global evaluation
        return None
    
    def _aggregate_fedavg(self, results: List[Tuple[Any, FitRes]]) -> Tuple[Parameters, Dict[str, Any]]:
        """Improved FedAvg aggregation with better parameter handling."""
        if not results:
            print("Warning: No results to aggregate!")
            return self.global_model, {}

        # Extract weights and sample sizes
        weights_list = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        sample_sizes = [fit_res.num_examples for _, fit_res in results]
        
        print(f"Aggregating {len(results)} client results with sample sizes: {sample_sizes}")
        
        # Verify all clients have same parameter structure
        if len(set(len(w) for w in weights_list)) > 1:
            print("Warning: Clients have different parameter counts!")
            param_counts = [len(w) for w in weights_list]
            print(f"Parameter counts: {param_counts}")
        
        # Perform weighted averaging
        total_samples = sum(sample_sizes)
        if total_samples == 0:
            print("Warning: Total samples is 0!")
            return self.global_model, {}
            
        aggregated_weights = []
        
        for i in range(len(weights_list[0])):
            # Compute weighted average for parameter i
            weighted_sum = np.zeros_like(weights_list[0][i])
            for weights, size in zip(weights_list, sample_sizes):
                weighted_sum += weights[i] * (size / total_samples)
            aggregated_weights.append(weighted_sum)
        
        # Verify aggregation didn't create NaN or inf values
        for i, param in enumerate(aggregated_weights):
            if np.isnan(param).any() or np.isinf(param).any():
                print(f"Warning: Parameter {i} contains NaN or inf values after aggregation!")
        
        # Update global model
        self.global_model = ndarrays_to_parameters(aggregated_weights)
        
        # Save the updated global model
        self._save_global_model(self.current_round)
        
        # Compute parameter change magnitude for debugging
        param_norms = [np.linalg.norm(p) for p in aggregated_weights]
        print(f"Aggregated parameter norms: {param_norms[:3]}...")  # Show first 3
        
        metrics = {
            "aggregation_type": "fedavg",
            "num_clients": len(results),
            "total_samples": total_samples,
            "param_norms": param_norms[:5]  # Store first 5 for debugging
        }
        
        return self.global_model, metrics
    
    def _aggregate_with_clustering(self, results: List[Tuple[Any, FitRes]]) -> Tuple[Parameters, Dict[str, Any]]:

        print(f"\n=== Clustering Phase (Round {self.current_round}) ===")
        
        global_params, _ = self._aggregate_fedavg(results)
        
        similarity_A, similarity_B = calculate_similarity(
            path=self.output_dir,
            n_clients=self.num_clients,
            round=self.current_round,
            layer='all'  # Using all layers for similarity

        )
        
        cluster_assignments = make_clusters(
            similarity_matrix=similarity_B,
            n_clusters=self.n_clusters,
            round=self.current_round,
            save_dendrogram=True,
            path=self.output_dir
            
        )
        
        self.cluster_assignments = cluster_assignments
        
        for client_idx in range(self.num_clients):
            self.client_clusters[client_idx] = cluster_assignments[client_idx] - 1
        
        print(f"Cluster assignments: {dict(self.client_clusters)}")

        cluster_models = self._create_cluster_models(results, cluster_assignments)
        self.cluster_models = cluster_models
        
        # Save cluster models
        self._save_cluster_models(self.current_round)
        
        metrics = {
            "aggregation_type": "clustering",
            "cluster_assignments": dict(self.client_clusters),
            "num_clusters": self.n_clusters,
            "similarity_calculated": True
        }
        
        return global_params, metrics
    
    def _aggregate_clustered(self, results: List[Tuple[Any, FitRes]]) -> Tuple[Parameters, Dict[str, Any]]:

        # Group results by cluster
        cluster_results = {}
        for client, fit_res in results:
            client_id = int(client.cid)
            cluster_id = self.client_clusters.get(client_id, 0)
            
            if cluster_id not in cluster_results:
                cluster_results[cluster_id] = []
            cluster_results[cluster_id].append((client, fit_res))
        
        # Aggregate within each cluster
        updated_cluster_models = {}
        cluster_metrics = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_res = cluster_results.get(cluster_id, [])
            
            if cluster_res:  # If there are clients in this cluster
                # Perform FedAvg within the cluster
                weights_list = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in cluster_res]
                sample_sizes = [fit_res.num_examples for _, fit_res in cluster_res]
                
                total_samples = sum(sample_sizes)
                if total_samples == 0:
                    print(f"Warning: Cluster {cluster_id} has 0 total samples!")
                    updated_cluster_models[cluster_id] = self.cluster_models.get(cluster_id, self.global_model)
                    continue
                    
                aggregated_weights = []
                
                for i in range(len(weights_list[0])):
                    weighted_sum = np.zeros_like(weights_list[0][i])
                    for weights, size in zip(weights_list, sample_sizes):
                        weighted_sum += weights[i] * (size / total_samples)
                    aggregated_weights.append(weighted_sum)
                
                updated_cluster_models[cluster_id] = ndarrays_to_parameters(aggregated_weights)
                cluster_metrics[cluster_id] = {
                    "num_clients": len(cluster_res),
                    "total_samples": total_samples
                }
                
                print(f"Cluster {cluster_id}: {len(cluster_res)} clients, {total_samples} samples")
            else:
                updated_cluster_models[cluster_id] = self.cluster_models.get(cluster_id, self.global_model)
                cluster_metrics[cluster_id] = {"num_clients": 0, "total_samples": 0}
                print(f"Cluster {cluster_id}: no clients, keeping previous model")
        
        # Update cluster models
        self.cluster_models.update(updated_cluster_models)
        
        # Save updated cluster models
        self._save_cluster_models(self.current_round)
        
        # Return the global model (could be any cluster model for backward compatibility)
        # In practice, this isn't used since clients get cluster-specific models
        representative_model = list(self.cluster_models.values())[0] if self.cluster_models else self.global_model
        
        metrics = {
            "aggregation_type": "clustered",
            "cluster_metrics": cluster_metrics,
            "active_clusters": len([c for c in cluster_results.values() if c])
        }
        
        return representative_model, metrics
    
    def _create_cluster_models(self, results: List[Tuple[Any, FitRes]], cluster_assignments: np.ndarray) -> Dict[int, Parameters]:
        
        cluster_models = {}
        
        # Group client results by cluster
        cluster_groups = {}
        for i, (client, fit_res) in enumerate(results):
            client_id = int(client.cid)
            if client_id < len(cluster_assignments):
                cluster_id = cluster_assignments[client_id] - 1  # Convert to 0-indexed
                
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(fit_res)
        
        # Create aggregated model for each cluster
        for cluster_id, cluster_results in cluster_groups.items():
            if cluster_results:
                # Aggregate within cluster
                weights_list = [parameters_to_ndarrays(fit_res.parameters) for fit_res in cluster_results]
                sample_sizes = [fit_res.num_examples for fit_res in cluster_results]
                
                total_samples = sum(sample_sizes)
                aggregated_weights = []
                
                for i in range(len(weights_list[0])):
                    weighted_sum = sum(weights[i] * size for weights, size in zip(weights_list, sample_sizes))
                    aggregated_weights.append(weighted_sum / total_samples)
                
                cluster_models[cluster_id] = ndarrays_to_parameters(aggregated_weights)
                print(f"Created model for cluster {cluster_id} with {len(cluster_results)} clients")
        
        # Fill in missing clusters with global model
        for cluster_id in range(self.n_clusters):
            if cluster_id not in cluster_models:
                cluster_models[cluster_id] = self.global_model
                print(f"Using global model for empty cluster {cluster_id}")
        
        return cluster_models


    def _save_initial_global_model(self) -> None:
        """Save the initial global model."""
        os.makedirs(os.path.join(self.output_dir, "global_model"), exist_ok=True)
        self._save_global_model(0)
        
    def _save_global_model(self, round_idx: int, peft_state_dict=None) -> None:
        """Save the global model for a specific round."""
        # Create directory for the round
        save_dir = os.path.join(self.output_dir, f"checkpoint-{round_idx}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Get parameters and reconstruct state dict with original keys
        params = parameters_to_ndarrays(self.global_model)
        if peft_state_dict is None:
            peft_state_dict = self.peft_state_dict
            
        # Recreate state dict with previous keys and updated parameters
        updated_state_dict = {}
        for i, (key, _) in enumerate(peft_state_dict.items()):
            updated_state_dict[key] = torch.tensor(params[i])
        
        # Save model using safetensors format
        save_file(updated_state_dict, os.path.join(save_dir, "adapter_model.safetensors"))
        print(f"Saved global model for round {round_idx}")
        
    def _save_cluster_models(self, round_idx: int, cluster_peft_state_dicts=None) -> None:
        """Save all cluster models for a specific round."""
        base_dir = os.path.join(self.output_dir, "cluster_models", f"round_{round_idx}")
        os.makedirs(base_dir, exist_ok=True)
        
        # Save each cluster model
        for cluster_id, cluster_params in self.cluster_models.items():
            cluster_dir = os.path.join(base_dir, f"cluster_{cluster_id}")
            os.makedirs(cluster_dir, exist_ok=True)
            
            # Get parameters for this cluster
            params = parameters_to_ndarrays(cluster_params)
            
            # Recreate state dict with previous keys and updated parameters
            updated_state_dict = {}
            for i, (key, _) in enumerate(self.peft_state_dict.items()):
                updated_state_dict[key] = torch.tensor(params[i])
            
            # Save model using safetensors format
            save_file(updated_state_dict, os.path.join(cluster_dir, "adapter_model.safetensors"))
            print(f"Saved model for cluster {cluster_id}, round {round_idx}")
            
    def _save_parameters_to_path(self, param_arrays: List[np.ndarray], output_path: str) -> None:
        """Save model parameters to a specific path."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Recreate state dict with original keys and provided parameters
        updated_state_dict = {}
        for i, (key, _) in enumerate(self.peft_state_dict.items()):
            if i < len(param_arrays):
                updated_state_dict[key] = torch.tensor(param_arrays[i])
        
        # Save using safetensors format
        save_file(updated_state_dict, output_path)
        print(f"Saved parameters to {output_path}")
       

    def _load_client_adapters(self, round_idx: int) -> Dict[int, List[np.ndarray]]:
        """
        Load client adapters from the output directory.
        
        Args:
            round_idx: Round to load client adapters from
            
        Returns:
            Dictionary mapping client_id to parameter arrays
        """
        client_adapters = {}
        adapter_dir = os.path.join(self.output_dir, "client_adapters", f"round_{round_idx}")
        
        if not os.path.exists(adapter_dir):
            print(f"No client adapters found for round {round_idx}")
            return client_adapters
        
        for client_folder in os.listdir(adapter_dir):
            if client_folder.startswith("client_"):
                try:
                    client_id = int(client_folder.split("_")[1])
                    client_path = os.path.join(adapter_dir, client_folder)
                    
                    # Try to load the adapter parameters
                    param_file = os.path.join(client_path, "model_parameters.npz")
                    if os.path.exists(param_file):
                        loaded = np.load(param_file)
                        param_arrays = [loaded[f"arr_{i}"] for i in range(len(loaded.files))]
                        client_adapters[client_id] = param_arrays
                        print(f"Loaded adapter for client {client_id}")
                    else:
                        # Fallback: try to load pytorch model
                        pytorch_file = os.path.join(client_path, "pytorch_model.bin")
                        if os.path.exists(pytorch_file):
                            state_dict = torch.load(pytorch_file, map_location="cpu")
                            param_arrays = [tensor.numpy() for tensor in state_dict.values()]
                            client_adapters[client_id] = param_arrays
                            print(f"Loaded adapter for client {client_id} from pytorch file")
                        else:
                            print(f"No parameter file found for client {client_id}")
                            
                except Exception as e:
                    print(f"Failed to load adapter for {client_folder}: {e}")
        
        return client_adapters


def create_server_strategy(experiment_config: Dict[str, Any]) -> ClusteredFedMLLMStrategy:

    initial_state_dict = get_peft_model_state_dict(experiment_config['model'])
    initial_parameters = ndarrays_to_parameters(
        [val.cpu().numpy() for val in initial_state_dict.values()]
    )

    peft_state_dict = get_peft_model_state_dict(experiment_config['model'])
    
    # Create strategy with clustering configuration
    strategy = ClusteredFedMLLMStrategy(
        initial_parameters=initial_parameters,
        peft_state_dict=peft_state_dict,
        num_clients=experiment_config['fed_args'].num_clients,
        sim_round=experiment_config['fed_args'].sim_round,
        n_clusters=experiment_config['fed_args'].n_clusters,
        output_dir=experiment_config['script_args'].output_dir,
        fraction_fit=1.0,  # Use all clients (can be adjusted)
        fraction_evaluate=1.0,
        min_fit_clients=experiment_config['fed_args'].num_clients,
        min_evaluate_clients=experiment_config['fed_args'].num_clients,
        min_available_clients=experiment_config['fed_args'].num_clients,
        script_args=experiment_config['script_args'],
        fed_args=experiment_config['fed_args']
    )
    
    return strategy
