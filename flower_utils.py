
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import DataCollatorForCompletionOnlyLM


def get_model_flower(script_args, training_args, peft_config,
                     device_map, quantization_config, torch_dtype):
    
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

    return model, tokenizer