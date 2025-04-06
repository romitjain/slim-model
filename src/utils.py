import torch
from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, List
import torch.utils.hooks

def load_model(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()

    return tokenizer, model

def remove_hooks(activation_handles: List[torch.utils.hooks.RemovableHandle], attention_handles: List[torch.utils.hooks.RemovableHandle]):
    for handle in activation_handles:
        handle.remove()
    for handle in attention_handles:
        handle.remove()

def prepare_dataset(dataset_name: str, config: str, split: str, tokenizer: AutoTokenizer) -> Dataset:
    dataset = load_dataset(dataset_name, config)[split]

    def prep_toks(x):
        msg = [
            {
                "role": "system",
                "content": "You are an helpful AI assistant whose job is to provide a concise summarize of the given content"
            },
            {
                "role": "user",
                "content": x["article"]
            }
        ]
        return {
            "tokens": tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        }

    dataset = dataset.select(range(128)).map(prep_toks)
    assert "tokens" in dataset.column_names
    return dataset


def prepare_train_dataset(dataset_name: str, config: str, split: str) -> Dataset:
    dataset = load_dataset(dataset_name, config)[split]

    def prep_toks(x):
        msg = [
            {
                "role": "system",
                "content": "You are an helpful AI assistant whose job is to provide a concise summarize of the given content"
            },
            {
                "role": "user",
                "content": x["article"]
            },
            {
                "role": "assistant",
                "content": x["summary"]
            }
        ]
        return {
            "messages": msg
        }

    dataset = dataset.map(prep_toks)
    assert "messages" in dataset.column_names

    return dataset
