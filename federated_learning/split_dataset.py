import random
import json
from datasets import concatenate_datasets

def split_dataset(fed_args, script_args, dataset, n_domains = 2):
    dataset = dataset.shuffle(seed=script_args.seed)        # Shuffle the dataset
    local_datasets = []
    if fed_args.split_strategy == "iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    
    #------------- IID datasets ----------------
    if fed_args.split_strategy == "language_iid":
        
        #languages = ['English', 'Swedish', 'German', 'Portuguese', 'Spanish']
        languages = ['English', 'Dutch', 'Turkish', 'Portuguese', 'Spanish']
        dataset = dataset.filter(lambda x: x['language'] in languages)
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    
    if fed_args.split_strategy == "ag_news_iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))

    if fed_args.split_strategy == "dolly_iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    
    if fed_args.split_strategy == "multitask_iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))

    #------------- Non-IID datasets ----------------

    ## Clustered (One Domain per Client)

    if fed_args.split_strategy == "language_clusters":
        #languages = ['English', 'Swedish', 'German', 'Portuguese', 'Spanish']
        languages = ['English', 'Dutch', 'Turkish', 'Portuguese', 'Spanish']
        n_clients_in_cluster = fed_args.num_clients // len(languages)

        for i in range(fed_args.num_clients):
            language = languages[i // n_clients_in_cluster]
            cluster_dataset = dataset.filter(lambda x: x['language'] == language)
            cluster_dataset = cluster_dataset.shuffle(seed=script_args.seed)

            local_datasets.append(cluster_dataset.shard(n_clients_in_cluster, i % n_clients_in_cluster))
    
    if fed_args.split_strategy == "multi_language_clusters":
        
        languages_high = ['English', 'Dutch', 'Turkish', 'Portuguese', 'Spanish']
        languages_mid = ['Filipino', 'Bengali', 'Standard Malay', 'Lithuanian', 'Tamil']
        languages_low = ['Zulu', 'Irish', 'Nepali', 'Malayalam', 'Telugu']
        languages = languages_high + languages_mid + languages_low

        n_clients_in_cluster = fed_args.num_clients // len(languages)

        for i in range(fed_args.num_clients):
            language = languages[i // n_clients_in_cluster]
            cluster_dataset = dataset.filter(lambda x: x['language'] == language)
            cluster_dataset = cluster_dataset.shuffle(seed=script_args.seed)

            local_datasets.append(cluster_dataset.shard(n_clients_in_cluster, i % n_clients_in_cluster))

    if fed_args.split_strategy == "ag_news_clusters":
        n_clients_in_cluster = fed_args.num_clients // 4
        categories = [0, 1, 2, 3] # 0: World, 1: Sports, 2: Business, 3: Sci/Tech

        for i in range(fed_args.num_clients):
            category = categories[i // n_clients_in_cluster]
            cluster_dataset = dataset.filter(lambda x: x['label'] == category)
            cluster_dataset = cluster_dataset.shuffle(seed=script_args.seed)

            local_datasets.append(cluster_dataset.shard(n_clients_in_cluster, i % n_clients_in_cluster))
    
    if fed_args.split_strategy == "dolly_clusters":
        tasks = ['open_qa', 'general_qa', 'closed_qa', 'classification', 'brainstorming', 'information_extraction', 'summarization']
        n_clients_in_cluster = fed_args.num_clients // len(tasks)

        for i in range(fed_args.num_clients):
            task = tasks[i // n_clients_in_cluster]
            cluster_dataset = dataset.filter(lambda x: x['category'] == task)
            cluster_dataset = cluster_dataset.shuffle(seed=script_args.seed)

            local_datasets.append(cluster_dataset.shard(n_clients_in_cluster, i % n_clients_in_cluster))
    
    if fed_args.split_strategy == "multitask_clusters":
        tasks = ['boolq', 'gigaword', 'samsum', 'webnlg']
        n_clients_in_cluster = fed_args.num_clients // len(tasks)

        for i in range(fed_args.num_clients):
            task = tasks[i // n_clients_in_cluster]
            cluster_dataset = dataset.filter(lambda x: x['task'] == task)
            cluster_dataset = cluster_dataset.shuffle(seed=script_args.seed)

            local_datasets.append(cluster_dataset.shard(n_clients_in_cluster, i % n_clients_in_cluster))

    ## Multi-Domain (2 Domains per Client)
    if fed_args.split_strategy == "language_multi_domain":
        #languages = ['English', 'Swedish', 'German', 'Portuguese', 'Spanish']
        languages = ['English', 'Dutch', 'Turkish', 'Portuguese', 'Spanish']
        n_clients_in_cluster = fed_args.num_clients // len(languages)

        for i in range(fed_args.num_clients):
            # Each client receives data from two languages
            lang1 = languages[i // n_clients_in_cluster]
            lang2 = languages[(i // n_clients_in_cluster + 1) % len(languages)]
            # Filter dataset for either of the two languages
            client_dataset_lang1 = dataset.filter(lambda x, lang1=lang1: x['language'] == lang1)
            client_dataset_lang2 = dataset.filter(lambda x, lang2=lang2: x['language'] == lang2)

            #get the fist 50% of the dataset from lang1 and the last 50% from lang2
            client_dataset_lang1 = client_dataset_lang1.select(range(len(client_dataset_lang1)//2))
            client_dataset_lang2 = client_dataset_lang2.select(range(len(client_dataset_lang2)//2, len(client_dataset_lang2)))
            client_dataset = concatenate_datasets([client_dataset_lang1, client_dataset_lang2])
        

            client_dataset = client_dataset.shuffle(seed=script_args.seed)
            local_datasets.append(client_dataset.shard(n_clients_in_cluster, i % n_clients_in_cluster))
        save_multi_domain_dataset_stats(local_datasets, script_args.output_dir)
        
    if fed_args.split_strategy == "ag_news_multi_domain":
        n_clients_in_cluster = fed_args.num_clients // 4
        categories = [0, 1, 2, 3]

        for i in range(fed_args.num_clients):
            # Each client receives data from two categories
            cat1 = categories[i // n_clients_in_cluster]
            cat2 = categories[(i // n_clients_in_cluster + 1) % len(categories)]
            # Filter dataset for either of the two categories
            client_dataset = dataset.filter(lambda x, cat1=cat1, cat2=cat2: x['label'] == cat1 or x['label'] == cat2)
            client_dataset = client_dataset.shuffle(seed=script_args.seed)
            local_datasets.append(client_dataset.shard(n_clients_in_cluster, i % n_clients_in_cluster))
        save_multi_domain_dataset_stats(local_datasets, script_args.output_dir)
    
    if fed_args.split_strategy == "dolly_multi_domain":
        tasks = ['open_qa', 'general_qa', 'closed_qa', 'classification', 'brainstorming', 'information_extraction', 'summarization']
        n_clients_in_cluster = fed_args.num_clients // len(tasks)

        for i in range(fed_args.num_clients):
            # Each client receives data from two tasks
            task1 = tasks[i // n_clients_in_cluster]
            task2 = tasks[(i // n_clients_in_cluster + 1) % len(tasks)]
            # Filter dataset for either of the two tasks
            client_dataset = dataset.filter(lambda x, task1=task1, task2=task2: x['category'] == task1 or x['category'] == task2)
            client_dataset = client_dataset.shuffle(seed=script_args.seed)
            local_datasets.append(client_dataset.shard(n_clients_in_cluster, i % n_clients_in_cluster))
        save_multi_domain_dataset_stats(local_datasets, script_args.output_dir)
        
    if fed_args.split_strategy == "multitask_multi_domain":
        tasks = ['boolq', 'gigaword', 'samsum', 'webnlg']
        n_clients_in_cluster = fed_args.num_clients // len(tasks)
        for i in range(fed_args.num_clients):
            # Each client receives data from two tasks
            task1 = tasks[i // n_clients_in_cluster]
            task2 = tasks[(i // n_clients_in_cluster + 1) % len(tasks)]
            # Filter dataset for either of the two tasks
            client_dataset_task1 = dataset.filter(lambda x, task1=task1: x['task'] == task1)
            client_dataset_task2 = dataset.filter(lambda x, task2=task2: x['task'] == task2)

            #get the fist 50% of the dataset from task1 and the last 50% from task2
            client_dataset_task1 = client_dataset_task1.select(range(len(client_dataset_task1)//2))
            client_dataset_task2 = client_dataset_task2.select(range(len(client_dataset_task2)//2, len(client_dataset_task2)))
            client_dataset = concatenate_datasets([client_dataset_task1, client_dataset_task2])
            #shuffle the dataset
            client_dataset = client_dataset.shuffle(seed=script_args.seed)

            local_datasets.append(client_dataset.shard(n_clients_in_cluster, i % n_clients_in_cluster))
    # Save dataset statistics
        save_multi_domain_dataset_stats(local_datasets, script_args.output_dir)

    save_dataset_stats(local_datasets, script_args.output_dir)
    return local_datasets


def save_multi_domain_dataset_stats(local_datasets, path):
    dataset_stats = {}
    for i, dataset in enumerate(local_datasets):
        domains = set()
        for sample in dataset:
            if 'language' in sample:
                domains.add(sample['language'])
            elif 'label' in sample:
                domains.add(sample['label'])
            elif 'task' in sample:
                domains.add(sample['task'])
        dataset_stats[f'client_{i}'] = list(domains)
    with open(path + '/multi_domain_dataset_stats.json', 'w') as f:
        json.dump(dataset_stats, f)

def save_dataset_stats(local_datasets, path):
    dataset_stats = {}
    for i, dataset in enumerate(local_datasets):
        dataset_stats[f'client_{i}'] = len(dataset)
    with open(path + '/dataset_stats.json', 'w') as f:
        json.dump(dataset_stats, f)

def get_dataset_this_round(dataset, round, fed_args, script_args):
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    num2sample = min(num2sample, len(dataset))
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)

    return dataset_this_round