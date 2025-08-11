import datasets
from datasets import load_dataset, concatenate_datasets
import pandas as pd
from .conversation import get_conv_template
from functools import partial

def get_dataset(dataset_name, local_data_dir=None, train_split = 1):

    if dataset_name in ["gsm8k"]:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train", name="main")
    elif dataset_name in ["lighteval/MATH"]:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train", name="all")
    elif dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train_sft")

    elif dataset_name in ["CohereForAI/aya_dataset"]:
        #languages = ['English', 'Swedish', 'German', 'Portuguese', 'Spanish']
        languages = ['English', 'Dutch', 'Turkish', 'Portuguese', 'Spanish']
        dataset = load_dataset(dataset_name, split="train")
        print('Start filtering languages')
        dataset = dataset.filter(lambda x: x['language'] in languages)

    elif dataset_name in ['multi_language_clusters']:

        languages_high = ['English', 'Dutch', 'Turkish', 'Portuguese', 'Spanish']
        languages_mid = ['Filipino', 'Bengali', 'Standard Malay', 'Lithuanian', 'Tamil']
        languages_low = ['Zulu', 'Irish', 'Nepali', 'Malayalam', 'Telugu']
        languages = languages_high + languages_mid + languages_low

        dataset = load_dataset("CohereForAI/aya_dataset", split="train")
        print('Start filtering languages')
        dataset = dataset.filter(lambda x: x['language'] in languages)
    
    elif dataset_name in ['fancyzhx/ag_news']:
        dataset = load_dataset(dataset_name, split="train")
    
    elif dataset_name in ["databricks/databricks-dolly-15k"]:
        dataset = load_dataset(dataset_name, split="train")

    elif dataset_name in ['multitask']:

        boolq = prepare_boolq(eval=eval).map(lambda x: {'instruction': x['instruction'], 'response': x['response'], 'task': 'boolq'})
        webnlg = prepare_webnlg(eval=eval).map(lambda x: {'instruction': x['instruction'], 'response': x['response'], 'task': 'webnlg'})
        samsum = prepare_samsum(eval=eval).map(lambda x: {'instruction': x['instruction'], 'response': x['response'], 'task': 'samsum'})
        gigaword = prepare_gigaword(eval=eval).map(lambda x: {'instruction': x['instruction'], 'response': x['response'], 'task': 'gigaword'})

        dataset = concatenate_datasets([boolq, webnlg, samsum, gigaword])
        dataset = dataset.shuffle(seed=0)

    else:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train")
    
    if train_split < 1:
            dataset_splited = dataset.train_test_split(test_size= 1-train_split, seed=0)
            dataset_train = dataset_splited['train']
            dataset_test = dataset_splited['test']
            print(f"Dataset Splited into TRAIN: {len(dataset_train)} and TEST: {len(dataset_test)}")
            return dataset_train, dataset_test

    return dataset

def process_sft_dataset(dataset_name, dataset, dataset_sample):
    if dataset_name in ["lucasmccabe-lmi/CodeAlpaca-20k", "yahma/alpaca-cleaned", "FinGPT/fingpt-sentiment-train"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
    elif dataset_name in ["WizardLM/WizardLM_evol_instruct_70k"]:
        dataset = dataset.rename_column("output", "response")
    elif dataset_name in ["tatsu-lab/alpaca", "vicgalle/alpaca-gpt4",
                          "gbharti/finance-alpaca"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output', 'text'], desc=f"Preprocessing {dataset_name} for unified format.")
    
    elif dataset_name in ["dominguesm/alpaca-data-pt-br"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
    
    elif dataset_name in ["CohereForAI/aya_dataset", 'multi_language_clusters']:
        dataset = dataset.map(alpaca_format_aya, remove_columns=['inputs', 'targets', 'language_code', 'annotation_type', 'user_id'],
                              desc=f"Preprocessing {dataset_name} for unified format.")

    elif dataset_name in ['fancyzhx/ag_news']:
        dataset = dataset
    
    elif dataset_name in ["databricks/databricks-dolly-15k"]:
        dataset = dataset.map(dolly_format, remove_columns=['context'], desc=f"Preprocessing {dataset_name} for unified format.")

    elif dataset_name in ['multitask']: #already formated on previous load step
        return dataset

    elif dataset_name in ["TIGER-Lab/MathInstruct"]:
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=['instruction'])
        dataset = datasets.Dataset.from_pandas(df)
        dataset = dataset.rename_column("output", "response")
        dataset = dataset.remove_columns(['source'])
    elif dataset_name in ["lighteval/MATH"]:
        dataset = dataset.rename_column("solution", "response")
        dataset = dataset.rename_column("problem", "instruction")
        dataset = dataset.remove_columns(['level', 'type'])
    elif dataset_name in ['gsm8k']:
        dataset = dataset.rename_column("question", "instruction")
        dataset = dataset.rename_column("answer", "response")
    elif dataset_name in ['medalpaca/medical_meadow_medical_flashcards']:       # TODO: 'lavita/ChatDoctor-HealthCareMagic-100k'. not sure whether to discard the instruction.
        dataset = dataset.remove_columns(['instruction'])
        dataset = dataset.rename_column("input", "instruction")
        dataset = dataset.rename_column("output", "response")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    dataset = dataset.shuffle(seed=2023)
    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    return dataset

def alpaca_format(example):
    if example['input'] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example['input']
    example["response"] = example['output']
    return example

def alpaca_format_aya(example):
    example["instruction"] = example['inputs']
    example["response"] = example['targets']

    return example

def dolly_format(example):
    if example['context'] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example['context']

    return example

def process_dpo_dataset(dataset_name, dataset, template_name, dataset_sample):
    if dataset_name in ["Anthropic/hh-rlhf"]:
        dataset = dataset.map(partial(split_hh, template_name=template_name), load_from_cache_file=False)
    elif dataset_name in ["HuggingFaceH4/ultrafeedback_binarized"]:
        dataset = dataset.map(partial(split_ultrafeedback, template_name=template_name), load_from_cache_file=False)
        dataset = dataset.remove_columns(['prompt_id', 'messages', 'score_chosen', 'score_rejected'])
    
    dataset = dataset.shuffle(seed=2023)
    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    print(f">> ===== Data Example =====")
    print(dataset[0])
    print(f">> {'='*50}")
    return dataset
    
def find_common_prefix(str1, str2):
    prefix = ""
    for i in range(min(len(str1), len(str2))):
        if str1[i] == str2[i]:
            prefix += str1[i]
        else:
            break
    return prefix

def split_ultrafeedback(example, template_name="vicuna_v1.1"):
    conv_template = get_conv_template(template_name)

    conv_template.append_message(conv_template.roles[0], example["prompt"])
    conv_template.append_message(conv_template.roles[1], None)
    example["prompt"] = conv_template.get_prompt()
    example["chosen"] = " " + example["chosen"][1]["content"]       # There might need a space in the front.
    example["rejected"] = " " + example["rejected"][1]["content"]
    return example

def split_hh(example, template_name="vicuna_v1.1"):
    common_prefix = find_common_prefix(example["chosen"], example["rejected"])

    conv_template = get_conv_template(template_name)

    sentence = common_prefix
    human_prefix_len = len("\n\nHuman: ")
    assistant_prefix_len = len("\n\nAssistant: ")
    sentence = sentence[human_prefix_len:]
    turn = "user"
    while True:
        if turn == "user":
            index = sentence.find("\n\nAssistant: ")
            if index == -1:
                break
            else:
                conv_template.append_message(conv_template.roles[0], sentence[:index])
                turn = "assistant"
                sentence = sentence[index + assistant_prefix_len :]
        elif turn == "assistant":
            index = sentence.find("\n\nHuman: ")
            if index == -1:
                break
            else:
                conv_template.append_message(conv_template.roles[1], sentence[:index])
                turn = "user"
                sentence = sentence[index + human_prefix_len :]
    conv_template.append_message(conv_template.roles[1], None)
    example["prompt"] = conv_template.get_prompt()
    example["chosen"] = example["chosen"][len(common_prefix) - 1 :]     # -1 to include the space in the front.
    example["rejected"] = example["rejected"][len(common_prefix) - 1 :]
    return example


##### FUNCTIONS FOR MULTITASK PROCESS

def prepare_webnlg(eval = False):
    dataset = load_dataset('GEM/web_nlg', 'en', split = 'train')
    #dataset = dataset.train_test_split(test_size=0.2, seed=0)

    # if eval:
    #     dataset = dataset['test']
    # else:
    #     dataset = dataset['train']
    
    dataset = dataset.map(webnlg_format)

    return dataset

def prepare_boolq(eval = False):
    dataset = load_dataset('google/boolq', split = 'train')
    #dataset = dataset.train_test_split(test_size=0.2, seed=0)

    # if eval:
    #     dataset = dataset['test']
    # else:
    #     dataset = dataset['train']
    
    dataset = dataset.map(boolq_format)

    return dataset

def prepare_samsum(eval = False):
    dataset = load_dataset('Samsung/samsum', split = 'train', trust_remote_code=True)
    #dataset = dataset.train_test_split(test_size=0.2, seed=0)

    # if eval:
    #     dataset = dataset['test']
    # else:
    #     dataset = dataset['train']

    dataset = dataset.map(samsum_format)

    return dataset

def prepare_gigaword(eval = False):
    dataset = load_dataset('Harvard/gigaword', split = 'train', trust_remote_code=True)
    #dataset = dataset.train_test_split(test_size=0.2, seed=0)

    # if eval:
    #     dataset = dataset['test']
    # else:
    #     dataset = dataset['train']
    
    dataset = dataset.shuffle(seed=0)
    dataset = dataset.select(range(30000)) #loading only part (the whole dataset has aroud 4M examples)
    dataset = dataset.map(gigaword_format)

    return dataset

def boolq_format(example):
    #example["instruction"] = example['passage'] + " Based on the passage, answer this question:" + example['question']
    example["instruction"] = example['passage'] + '-' + example['question']
    example["response"] = str(example['answer'])
    return example

def webnlg_format(example):
    example['input'] = str(example['input'])
    #example["instruction"] = "Organize this data into a readable text: " + example['input']
    example["instruction"] = example['input']
    example["response"] = example['target']
    return example

def samsum_format(example):
    #example["instruction"] = "Summarize this conversation: " + example['dialogue']
    example["instruction"] = example['dialogue']
    example["response"] = example['summary']
    return example

def gigaword_format(example):
    #example["instruction"] = "Summarize this text: " + example['document']
    example["instruction"] = example['document']
    example["response"] = example['summary']
    return example