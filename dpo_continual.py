from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from dataclasses import dataclass
import tyro
import torch
import random
import numpy as np

import wandb


@dataclass
class Args:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct" # facebook/opt-350m gpt2 google/flan-t5-small
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    seed: int = 1
    """seed of the experiment"""
    wandb_project_name: str = "conitnual_finetuning_baselines"
    """the wandb's project name"""
    wandb_entity: str = 'irina-rish'
    """the entity (team) of wandb's project"""
    num_train_epochs: int = 1
    """number of training epochs"""
    per_device_train_batch_size: int = 4
    """batch size per device"""


if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    group_name = f"{args.model_name}-DPO"

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name, peft_config=lora_config,)

    model.warnings_issued = {}
    model.config.return_dict = True

    # Apply LoRA to the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set the tokenizer's pad_token to eos_token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ToDo: replace with our actual datasets, define the eval datasets as well
    datasets = [
        load_dataset("trl-lib/ultrafeedback_binarized", split="test"),
        load_dataset("Anthropic/hh-rlhf", split="test")
                ]

    training_args = DPOConfig(output_dir=group_name)
    training_args.per_device_train_batch_size = args.per_device_train_batch_size
    training_args.num_train_epochs = args.num_train_epochs

    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, group=group_name, name=group_name, config=vars(training_args))

    for i, dataset in enumerate(datasets):
        # ToDo: log the dataset name or index, that should come from the dataset itself
        wandb.log({"dataset": i})
        trainer = DPOTrainer(model=model, args=training_args, train_dataset=dataset, eval_dataset=dataset, processing_class=tokenizer)
        trainer.train()