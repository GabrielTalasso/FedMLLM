import random
import numpy as np
import torch
from federated_learning.fed_clustered import calculate_similarity, make_clusters, calculate_similarity_pair
from federated_learning.fed_personalized import get_adapter

def get_clients_this_round(fed_args, script_args, round):
    if (fed_args.fed_alg).startswith('local'):
        clients_this_round = [int((fed_args.fed_alg)[-1])]
    else:
        if fed_args.num_clients < fed_args.sample_clients:
            clients_this_round = list(range(fed_args.num_clients))
        else:
            random.seed(script_args.seed + round)
            clients_this_round = sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))
    return clients_this_round

def global_aggregate(fed_args, script_args, global_dict, local_dict_list,
                     sample_num_list, clients_this_round, round_idx,
                     proxy_dict=None, opt_proxy_dict=None, auxiliary_info=None,
                     round = None, idx = None):
    
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None

    if fed_args.fed_alg == 'scaffold':
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        global_auxiliary, auxiliary_delta_dict = auxiliary_info
        for key in global_auxiliary.keys():
            delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in clients_this_round]) 
            global_auxiliary[key] += delta_auxiliary / fed_args.num_clients
    
    elif fed_args.fed_alg == 'fedavgm':
        # Momentum-based FedAvg
        for key in global_dict.keys():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            global_dict[key] = global_dict[key] + proxy_dict[key]

    elif fed_args.fed_alg == 'fedadagrad':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            # In paper 'adaptive federated optimization', momentum is not used
            proxy_dict[key] = delta_w
            opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedyogi':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            delta_square = torch.square(proxy_dict[key])
            opt_proxy_dict[key] = param - (1-fed_args.fedopt_beta2)*delta_square*torch.sign(param - delta_square)
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedadam':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            opt_proxy_dict[key] = fed_args.fedopt_beta2*param + (1-fed_args.fedopt_beta2)*torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg in ['clustered', 'clustered_random']:

        if round < fed_args.sim_round: # Normal dataset-size-based aggregation 
            for key in global_dict.keys():
              global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        
        elif round == fed_args.sim_round:
            
            n_clusters = fed_args.n_clusters

            # Calculate similarity matrix between clients adapters ----------------

            if n_clusters == fed_args.num_clients:
                pass
            else:
                similarity_A, similarity_B = calculate_similarity(path = script_args.output_dir,
                                                                n_clients = fed_args.num_clients,
                                                                round = round)
                                                                
            # Make clusters using hierarchical clustering ------------------------
            if fed_args.fed_alg == 'clustered':

                if n_clusters == fed_args.num_clients: #create one cluster for each clients (completely distributed)
                    idx = np.arange(1, n_clusters+1)
                else:
                    idx = make_clusters(similarity_matrix = similarity_B,
                                n_clusters = n_clusters,
                                round = round,
                                save_dendrogram = True, 
                                path = script_args.output_dir)
            
            elif fed_args.fed_alg == 'clustered_random':
                idx = np.random.randint(1, n_clusters+1, size=fed_args.num_clients)
                with open(script_args.output_dir + '/idx.txt', 'w') as f:
                    f.write(str(idx))
            
            # Separate models into clusters -------------------------------------
            clusters_models = {}
            for cluster in range(n_clusters):

                if cluster not in clusters_models.keys():
                    clusters_models[cluster] = []
                clusters_models[cluster].append([local_dict_list[client] for client in list(range(fed_args.num_clients)) if idx[client] == cluster+1])

            # Aggregate models within each cluster ------------------------------
            cluster_agg_models = [] 

            for cluster in range(n_clusters):
                cluster_dict = {}
                for key in global_dict.keys():
                    #print(key, cluster, len(clusters_models[cluster]), len(clusters_models[cluster][0]), len(clusters_models[cluster][0][0]))
                    cluster_dict[key] = sum([model[key] for model in clusters_models[cluster][0]]) / len(clusters_models[cluster][0])
                cluster_agg_models.append(cluster_dict)

            return cluster_agg_models, global_auxiliary, idx
        
        elif round > fed_args.sim_round:

            n_clusters = fed_args.n_clusters

            # Separate models into clusters -------------------------------------
            clusters_models = {}
            for cluster in range(n_clusters):

                if cluster not in clusters_models.keys():
                    clusters_models[cluster] = []
                clusters_models[cluster].append([local_dict_list[client] for client in clients_this_round if idx[client] == cluster+1]) #aggregate only selected clients

                if clusters_models[cluster] == [[]]:
                    clusters_models[cluster] = []
                    clusters_models[cluster].append([global_dict[cluster]]) #if no client selected in that cluster, use the previous global model

            # Aggregate models within each cluster ------------------------------
            cluster_agg_models = [] 

            for cluster in range(n_clusters):
                cluster_dict = {}
                for key in global_dict[cluster].keys():
                    cluster_dict[key] = sum([model[key] for model in clusters_models[cluster][0]]) / len(clusters_models[cluster][0])
                cluster_agg_models.append(cluster_dict)

            # Aggregate only the A LoRA adapter across clusters, keep B per cluster
            # find all A-keys
            #a_keys = [k for k in cluster_agg_models[0].keys() if 'lora_A' in k]
            ## compute cross-cluster average for A
            #aggregated_A = {
            #    k: sum(c[k] for c in cluster_agg_models) / len(cluster_agg_models)
            #    for k in a_keys
            #}
            ## replace A in each cluster model
            #for c_dict in cluster_agg_models:
            #    for k, v in aggregated_A.items():
            #        c_dict[k] = v

            return cluster_agg_models, global_auxiliary, idx

    elif fed_args.fed_alg in ['router', 'router_oracle']:

        n_clusters = fed_args.global_n_clusters

        # Separate models into clusters -------------------------------------
        clusters_models = {}
        for cluster in range(n_clusters):
            print(f'#Index returned: {len(idx)}, Models returned: {len(local_dict_list)}')
            print(clusters_models.keys())
            print(idx)
            if cluster not in clusters_models.keys():
                clusters_models[cluster] = []
            clusters_models[cluster].append([local_dict_list[center] for center in range(len(local_dict_list)) if idx[center] == cluster]) #aggregate only selected clients

            if clusters_models[cluster] == [[]]:
                clusters_models[cluster] = []
                clusters_models[cluster].append([global_dict[cluster]]) #if no client selected in that cluster, use the previous global model

        # Aggregate models within each cluster ------------------------------
        cluster_agg_models = [] 

        for cluster in range(n_clusters):
            cluster_dict = {}
            if round == 0:
                for key in global_dict.keys():
                    cluster_dict[key] = sum([model[key] for model in clusters_models[cluster][0]]) / len(clusters_models[cluster][0])
            else:
                for key in global_dict[cluster].keys():
                    cluster_dict[key] = sum([model[key] for model in clusters_models[cluster][0]]) / len(clusters_models[cluster][0])
            cluster_agg_models.append(cluster_dict)

        # Aggregate only the A LoRA adapter across clusters, keep B per cluster
        # find all A-keys
        #for cluster in range(n_clusters):
        #    a_keys = [k for k in cluster_agg_models[cluster].keys() if 'lora_A' in k]
        #    # compute cross-cluster average for A
        #    aggregated_A = {
        #        k: sum(c[k] for c in cluster_agg_models) / len(cluster_agg_models)
        #        for k in a_keys
        #    }
        #    # replace A in each cluster model
        #    for c_dict in cluster_agg_models:
        #        for k, v in aggregated_A.items():
        #            c_dict[k] = v

        return cluster_agg_models, global_auxiliary, idx

    elif fed_args.fed_alg == 'MTL':

        if round < fed_args.sim_round: # Normal dataset-size-based aggregation 
            for key in global_dict.keys():
              global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        
        elif round >= fed_args.sim_round:
            
            #Calculating a global model for create the delta of weights this round
            aggregate_all_models = {}
            if round == fed_args.sim_round:
                for key in global_dict.keys():
                    aggregate_all_models[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
            else:
                for key in global_dict[0].keys():
                    aggregate_all_models[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])

            #calcule similarity based on the previous differences
            similarity_A, similarity_B = calculate_similarity(path = script_args.output_dir,
                                                            n_clients = fed_args.num_clients,
                                                            round = round,
                                                            adapter_global=aggregate_all_models)
            
            #aggregate the models based on the similarity (MTL)
            personalized_models = []
            for client in clients_this_round:

                personalized_model = {}
                weights = similarity_B[client] / sum(similarity_B[client])  # Normalize weights

                if round == fed_args.sim_round:
                    # Initialize with global model
                    for key in global_dict.keys():
                        if 'lora_A' in key:
                            personalized_model[key] = sum([weights[i] * local_dict_list[i][key] for i in range(fed_args.num_clients)])
                        elif 'lora_B' in key:
                            personalized_model[key] = sum([weights[i] * local_dict_list[i][key] for i in range(fed_args.num_clients)])
                        
                    personalized_models.append(personalized_model)

                else:
                    for key in global_dict[0].keys():
                        if 'lora_A' in key:
                            personalized_model[key] = sum([weights[i] * local_dict_list[i][key] for i in range(fed_args.num_clients)])
                        elif 'lora_B' in key:
                            personalized_model[key] = sum([weights[i] * local_dict_list[i][key] for i in range(fed_args.num_clients)])
                    personalized_models.append(personalized_model)

            idx = np.array(clients_this_round) + np.ones_like(clients_this_round)
            return personalized_models, global_auxiliary, idx

    elif fed_args.fed_alg == 'FedSA':
        # Federated Shared A: Aggregate A matrix and maintain B matrices per client. Returns a personalized model for each client.
        
        global_dict = global_dict[0] if isinstance(global_dict, list) else global_dict  # Ensure global_dict is a single dict

        aggregated_A = {}
        for key in global_dict.keys():
            if 'lora_A' in key:
                aggregated_A[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        personalized_models = []

        for client in clients_this_round:
            personalized_model = {}
            for key in global_dict.keys():
                if 'lora_A' in key:
                    personalized_model[key] = aggregated_A[key]
                elif 'lora_B' in key:
                    personalized_model[key] = local_dict_list[client][key]
            personalized_models.append(personalized_model)

        idx = np.array(clients_this_round) + np.ones_like(clients_this_round)
        return personalized_models, global_auxiliary, idx

    elif fed_args.fed_alg == 'personalized':
        pass

        #not implemented yet =(
                
    else:   # Normal dataset-size-based aggregation 
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
    
    return global_dict, global_auxiliary