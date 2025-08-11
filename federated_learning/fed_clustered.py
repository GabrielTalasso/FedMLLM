import torch
import sys
import os
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as spc
from safetensors.torch import load_file

sys.path.append(".")


def get_adapter(path, client, round = 50, layer = -1, adapter_global = None):
    
    try:
        adapter_path_client = path + f'/clients_adapters/checkpoint-{round}_client{client}/adapter_model.bin'
        adapter_path_global = path+ f'/checkpoint-{round-1}/adapter_model.bin'

        # Load the adapter weights from the checkpoint
        adapter_state_dict = torch.load(adapter_path_client, map_location='cpu')
        adapter_global = torch.load(adapter_path_global, map_location='cpu')
    except FileNotFoundError:
        adapter_path_client = path + f'/clients_adapters/checkpoint-{round}_client{client}/adapter_model.safetensors'
        adapter_path_global = path+ f'/checkpoint-{round-1}/adapter_model.safetensors'

        # Load the adapter weights from the checkpoint using safetensors
        adapter_state_dict = load_file(adapter_path_client, device='cpu')

        if adapter_global is None:
            adapter_global = load_file(adapter_path_global, device='cpu')
        else:
            #to cpu
            adapter_global = {k: v.cpu() for k, v in adapter_global.items()}
            
        #adapter_global = torch.load(adapter_path_global, map_location='cpu', weights_only=False)


    # Access the adapter weights (as tensors)
    adapter_weights_A = [param for name, param in adapter_state_dict.items() if 'lora_A' in name]
    adapter_weights_B = [param for name, param in adapter_state_dict.items() if 'lora_B' in name]

    adapter_weights_A_global = [param for name, param in adapter_global.items() if 'lora_A' in name]
    adapter_weights_B_global = [param for name, param in adapter_global.items() if 'lora_B' in name]

    # Subtract the global adapter weights from the client adapter weights

    if layer == 'all':
        adapter_weights_A = [adapter_weights_A[i] - adapter_weights_A_global[i] for i in range(len(adapter_weights_A))]
        adapter_weights_B = [adapter_weights_B[i] - adapter_weights_B_global[i] for i in range(len(adapter_weights_B))]

        adapter_weights_A = torch.cat([adapter_weights_A[i].flatten() for i in range(len(adapter_weights_A))])
        adapter_weights_B = torch.cat([adapter_weights_B[i].flatten() for i in range(len(adapter_weights_B))])

    else:
        adapter_weights_A = adapter_weights_A[layer] - adapter_weights_A_global[layer]
        adapter_weights_B = adapter_weights_B[layer] - adapter_weights_B_global[layer]

    #adapter_weights_A = adapter_weights_A[layer]
    #adapter_weights_B = adapter_weights_B[layer]

    #flatten
    adapter_weights_A = adapter_weights_A.flatten()
    adapter_weights_B = adapter_weights_B.flatten()

    return adapter_weights_A,  adapter_weights_B

def calculate_similarity(path, n_clients, round, layer = -1, adapter_global = None):

    similarity_A = np.zeros((n_clients,n_clients))
    similarity_B = np.zeros((n_clients,n_clients))

    for c1 in list(range(n_clients)):
        #print(f'Calculating similarity for {c1}')
        adapter_weights_A_c1, adapter_weights_B_c1 = get_adapter(path, client = c1, round = round, layer = layer, adapter_global = adapter_global) 
        adapter_weights_A_c1 = adapter_weights_A_c1.cpu()
        adapter_weights_B_c1 = adapter_weights_B_c1.cpu()

        for c2 in list(range(n_clients)):
            adapter_weights_A_c2, adapter_weights_B_c2 = get_adapter(path, client = c2, round = round, layer = layer, adapter_global = adapter_global)
            adapter_weights_A_c2 = adapter_weights_A_c2.cpu()
            adapter_weights_B_c2 = adapter_weights_B_c2.cpu()
            
            #cosine similarity
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            cos_A = cos(adapter_weights_A_c1, adapter_weights_A_c2)
            cos_B = cos(adapter_weights_B_c1, adapter_weights_B_c2)
            similarity_A[c1][c2] = cos_A
            similarity_B[c1][c2] = cos_B
    
    #save matrices
    np.save(os.path.join(path, f"similarity_A_round{round}.npy"), similarity_A)
    np.save(os.path.join(path, f"similarity_B_round{round}.npy"), similarity_B)

    return similarity_A, similarity_B

def calculate_similarity_pair(adapter1, adapter2):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_sim = cos(adapter1, adapter2)

    return cos_sim

def make_clusters(similarity_matrix, n_clusters, round, save_dendrogram = True, path = None):

    pdist = spc.distance.pdist(similarity_matrix)
    linkage = spc.linkage(pdist, method='ward')
    min_link = linkage[0][2]
    max_link = linkage[-1][2]


    th = max_link
    for i in np.linspace(min_link,max_link, 5000):
        le = len(pd.Series(spc.fcluster(linkage, i, 'distance')).unique())
        if le == n_clusters:
            th = i

    idx = spc.fcluster(linkage, th, 'distance')
    print('Clusters created: ', idx)

    #save clusters
    np.save(os.path.join(path, f"clusters_round{round}.npy"), idx)

    if save_dendrogram:
        plt.figure(figsize=(10, 7))
        plt.title("Dendrogram")
        dendrogram(linkage)
        plt.savefig(os.path.join(path, f"dendrogram_round{round}.png"))

    return idx

