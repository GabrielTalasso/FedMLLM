import torch
import sys
import os
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as spc

sys.path.append(".")


def get_adapter(path, client, round = 50, layer = -1):

    adapter_path_client = path + f'/clients_adapters/checkpoint-{round}_client{client}/adapter_model.bin'
    adapter_path_global = path+ f'/checkpoint-{round-1}/adapter_model.bin'

    # Load the adapter weights from the checkpoint
    adapter_state_dict = torch.load(adapter_path_client, map_location='cpu')
    adapter_global = torch.load(adapter_path_global, map_location='cpu')

    # Access the adapter weights (as tensors)
    adapter_weights_A = [param for name, param in adapter_state_dict.items() if 'lora_A' in name]
    adapter_weights_B = [param for name, param in adapter_state_dict.items() if 'lora_B' in name]

    # Subtract the global adapter weights from the client adapter weights

    if layer == 'all':
        return adapter_weights_A,  adapter_weights_B
    
    else:
        return adapter_weights_A[layer],  adapter_weights_B[layer]