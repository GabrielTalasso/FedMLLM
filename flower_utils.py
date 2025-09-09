from functools import lru_cache
from typing import Any, Dict
import os
import copy
import numpy as np
import torch

from flwr.common import Context
from flwr.server import ServerConfig
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling,
)

# Import federated learning utilities
from utils.utils import cosine_learning_rate, default_evaluation, save_dataset_test
from federated_learning.split_dataset import get_dataset_this_round
from federated_learning.fed_local_sft import get_fed_local_sft_trainer
from federated_learning.fed_clustered import calculate_similarity, make_clusters


# ===== Builders =====
# ===== Model Builder =====
class ModelBuilder:
    def __init__(self):
        self._model_name: str = ""
        self._use_lora: bool = False
        self._lora_rank: int = 8

    def with_model_name(self, model_name: str) -> "ModelBuilder":
        """Define qual modelo carregar."""
        self._model_name = model_name
        return self

    def enable_lora(self, flag: bool = True) -> "ModelBuilder":
        """Ativa ou desativa LoRA."""
        self._use_lora = flag
        return self

    def with_lora_rank(self, rank: int) -> "ModelBuilder":
        """Define o rank de LoRA (r)."""
        self._lora_rank = rank
        return self

    def build(self):
        """Efetivamente carrega o modelo e (se pedido) aplica LoRA."""
        if not self._model_name:
            raise ValueError("Você deve chamar .with_model_name() antes de build()")

        # 1) Escolhe a família de modelo
        lname = self._model_name.lower()
        if "bert" in lname:
            model = AutoModelForMaskedLM.from_pretrained(self._model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(self._model_name)

        # 2) Se LoRA estiver habilitado, aplica-o
        if self._use_lora:
            lora_cfg = LoraConfig(
                r=self._lora_rank,
                lora_alpha=self._lora_rank * 2,
                lora_dropout=0.1,
            )
            model = get_peft_model(model, lora_cfg)

        return model


# ===== ServerConfig Builder =====
class ServerConfigBuilder:
    def __init__(self):
        self._num_rounds = 1

    def with_rounds(self, n: int) -> "ServerConfigBuilder":
        """Define o número de rounds do servidor."""
        self._num_rounds = n
        return self

    def build(self) -> ServerConfig:
        """Constroi e retorna a configuração do servidor."""
        return ServerConfig(num_rounds=self._num_rounds)


# ===== Tokenizer Builder =====
class TokenizerBuilder:
    def __init__(self):
        self._model_name: str = ""

    def with_model_name(self, model_name: str) -> "TokenizerBuilder":
        """Define qual modelo carregar."""
        self._model_name = model_name
        return self

    def build(self):
        """
        Carrega e retorna um AutoTokenizer configurado:
        - modelos BERT (contêm 'bert' no nome) sem padding.
        - demais modelos com padding=True e pad_token = eos_token.
        """
        name_lower = self._model_name.lower()
        if "bert" in name_lower:
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                use_fast=True,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                padding=True,
                use_fast=True,
            )
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


# ===== Traning Config Builder =====
class TraningConfigBuilder:
    def __init__(self):
        self.output_dir: str = ""
        self.logging_dir = ""
        self.logging_steps = 111
        self.learning_rate = 1e-3
        self.weight_decay = 0.01
        self.max_steps = 100
        self.num_train_epochs = 1
        self.save_steps = 1000
        self.eval_strategy = "steps"
        self.eval_steps = 111
        self.fp16 = True
        self.optim = "paged_adamw_8bit"
        self.lr_scheduler_type = "constant"

    def with_output_dir(self, output_dir: str) -> "TraningConfigBuilder":
        """Define qual modelo carregar."""
        self.output_dir = output_dir
        return self

    def with_logging_dir(self, logging_dir: str) -> "TraningConfigBuilder":
        self.logging_dir = logging_dir
        return self

    def with_logging_steps(self, logging_steps: int) -> "TraningConfigBuilder":
        self.logging_steps = logging_steps
        return self

    def with_learning_rate(self, learning_rate: float) -> "TraningConfigBuilder":
        self.learning_rate = learning_rate
        return self

    def with_weight_decay(self, weight_decay: float) -> "TraningConfigBuilder":
        self.weight_decay = weight_decay
        return self

    def with_max_steps(self, max_steps: int) -> "TraningConfigBuilder":
        self.max_steps = max_steps
        return self

    def with_num_train_epochs(self, num_train_epochs: int) -> "TraningConfigBuilder":
        self.num_train_epochs = num_train_epochs
        return self

    def with_save_steps(self, save_steps: int) -> "TraningConfigBuilder":
        self.save_steps = save_steps
        return self

    def with_eval_strategy(self, eval_strategy: str) -> "TraningConfigBuilder":
        self.eval_strategy = eval_strategy
        return self

    def with_eval_steps(self, eval_steps: int) -> "TraningConfigBuilder":
        self.eval_steps = eval_steps
        return self

    def with_fp16(self, fp16: bool) -> "TraningConfigBuilder":
        self.fp16 = fp16
        return self

    def with_optim(self, optim: str) -> "TraningConfigBuilder":
        self.optim = optim
        return self

    def with_lr_scheduler_type(self, lr_scheduler_type: str) -> "TraningConfigBuilder":
        self.lr_scheduler_type = lr_scheduler_type
        return self

    def build(self):
        return TrainingArguments(output_dir=f"{self.output_dir}/fl-results", logging_dir=f"{self.logging_dir}/logs",
                                 logging_steps=self.logging_steps, learning_rate=self.learning_rate,
                                 weight_decay=self.weight_decay, max_steps=self.max_steps,
                                 num_train_epochs=self.num_train_epochs, save_steps=self.save_steps,
                                 eval_strategy=self.eval_strategy, eval_steps=self.eval_steps, fp16=self.fp16,
                                 optim=self.optim, lr_scheduler_type=self.lr_scheduler_type)


# ===== Trainer Builder =====
class TrainerBuilder:
    def __init__(self):
        self.cid = 0
        self.model = None
        self.args = None
        self.train_dataset = None
        self.tokenizer = None
        self.eval_dataset = None
        self.model_name = ""

    def with_cid(self, cid) -> "TrainerBuilder":
        self.cid = cid
        return self

    def with_model(self, model) -> "TrainerBuilder":
        self.model = model
        return self

    def with_args(self, args) -> "TrainerBuilder":
        self.args = args
        return self

    def with_train_dataset(self, train_dataset) -> "TrainerBuilder":
        self.train_dataset = train_dataset
        return self

    def with_tokenizer(self, tokenizer) -> "TrainerBuilder":
        self.tokenizer = tokenizer
        return self

    def with_eval_dataset(self, eval_dataset) -> "TrainerBuilder":
        self.eval_dataset = eval_dataset
        return self

    def with_model_name(self, model_name) -> "TrainerBuilder":
        self.model_name = model_name
        return self

    def build(self):
        """Build and return a federated SFT trainer."""
        # Use the federated learning trainer builder
        trainer = get_fed_local_sft_trainer(
            script_args=None,  # Will be set later in actual usage
            fed_args=None,     # Will be set later in actual usage  
            model=self.model,
            tokenizer=self.tokenizer,
            training_args=self.args,
            local_dataset=self.train_dataset,
            formatting_prompts_func=None,  # Will be set later
            data_collator=self.data_collator,
            global_dict=None,
            local_auxiliary=None,
            global_auxiliary=None,
            packing=True
        )
        return trainer


# ===== Singletons =====
# ===== Tokenizer Singleton =====
@lru_cache(maxsize=1)
def get_tokenizer(model_name: str) -> Any:
    """Retorna sempre a mesma instância de AutoTokenizer para um dado model_name."""
    builder = TokenizerBuilder().with_model_name(model_name)
    return builder.build()


# ===== Factories =====
# ===== on_fit_config_fn Factory =====
class FitConfigFactory:
    def __init__(self, context: Context):
        # extrai e atribui apenas uma vez
        self.num_rounds = context.run_config["num-rounds"]
        self.initial_lr = context.run_config["initial-lr"]
        self.min_lr = context.run_config["min-lr"]
        self.dataset_path = context.run_config["dataset-path"]
        self.results_path = context.run_config["results-path"]
        self.model_name = context.run_config["model-name"]
        self.lora = context.run_config["lora"]

    def __call__(self, server_round: int) -> Dict[str, Any]:
        # devolve o dict completo ao FL
        return {
            "current_round": server_round,
            "num_rounds": self.num_rounds,
            "initial_lr": self.initial_lr,
            "min_lr": self.min_lr,
            "dataset_path": self.dataset_path,
            "results_path": self.results_path,
            "model_name": self.model_name,
            "lora": self.lora,
        }