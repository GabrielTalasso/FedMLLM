from transformers import AutoModelForCausalLM
from config import get_model_config
from sklearn.cluster import KMeans
import torch
from transformers import AutoTokenizer
import datasets
import numpy as np

def get_embeddings_model(texts, model, tokenizer):
    with torch.no_grad():
        inputs = tokenizer(
            texts, 
            return_tensors='pt', 
            padding='longest', 
            truncation=True, 
            max_length=1024,
            padding_side='left'
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(inputs['input_ids'], output_hidden_states=True, attention_mask=inputs['attention_mask'])
        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        attention_mask = inputs['attention_mask'].unsqueeze(-1)

        # Apply mask to zero out padding tokens
        masked_hidden_states = last_hidden_state * attention_mask

        # Sum and divide by the number of actual tokens (not padding)
        sum_hidden_states = masked_hidden_states.sum(dim=1)
        token_counts = attention_mask.sum(dim=1)

        # Avoid division by zero
        token_counts = torch.clamp(token_counts, min=1.0)

        # Calculate average excluding padding tokens
        embeddings = sum_hidden_states / token_counts
    return embeddings

def get_client_embedding(script_args, fed_args, client_dataset, batch_size=32):
    device_map, quantization_config, torch_dtype = get_model_config(script_args)
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        #quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        #torch_dtype=torch_dtype
    )
        
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, 
        use_fast=False, 
        padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   # following vicuna

    if tokenizer.eos_token == tokenizer.unk_token or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        #print(f"Pad token is set to {tokenizer.pad_token}.")

    #print('Special tokens:', tokenizer.special_tokens_map)
    model.resize_token_embeddings(len(tokenizer))

    texts = client_dataset['instruction']
    embeddings_list = []
    
    # Process the texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        batch_embeddings = get_embeddings_model(batch_texts, model, tokenizer)
        # Convert to CPU numpy array with float16 precision
        batch_embeddings = batch_embeddings.to(torch.float16).cpu().numpy()
        embeddings_list.append(batch_embeddings)
        
    embeddings = np.concatenate(embeddings_list, axis=0)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings

def cluster_embeddings(embeddings, num_clusters = 1):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
    kmeans.fit(embeddings)
    return kmeans.cluster_centers_, kmeans.labels_

def separate_data_into_clusters(sub_dataset, labels):
    clusters_datasets = []
    sub_dataset = sub_dataset.add_column('index', list(range(len(sub_dataset))))
    for i in range(max(labels) + 1):
        cluster_dataset = sub_dataset.filter(lambda x: labels[x['index']] == i)
        clusters_datasets.append(cluster_dataset)
    return clusters_datasets

def clusterize_dataset(embeddings, centroids):

    infered_labels = []

    for e in embeddings:
        #e = e.reshape(1, -1)
        distances = np.linalg.norm(centroids - e, axis=1)
        closest_centroid = np.argmin(distances)
        infered_labels.append(closest_centroid)
    
    return np.array(infered_labels)
    
def cluster_clients_centroids(client_embeddings_centers, num_clusters = 1):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
    kmeans.fit(client_embeddings_centers)
    return kmeans.cluster_centers_, kmeans.labels_

def get_most_similar_adapter(global_centroids, global_clusters, client_centroid):
    min_distance = float('inf')
    
    for i, centroid in enumerate(global_centroids):
        distance = np.linalg.norm(centroid - client_centroid)
        if distance < min_distance:
            min_distance = distance
            most_similar_adapter = i
            
    return most_similar_adapter


    

    