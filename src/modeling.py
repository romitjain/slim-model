# Load the pruned model and infer

import pdb
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2Config, Qwen2Model, Qwen2ForCausalLM


class PrunedQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx

        delattr(self, 'self_attn')
        delattr(self, 'post_attention_layernorm')

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        return (hidden_states,)


class PrunedQwen2Model(Qwen2Model):
    def __init__(self, config: Qwen2Config, pruned_layers: List[int]):
        super().__init__(config)
        self.pruned_layers = pruned_layers

        for layer_idx in self.pruned_layers:
            self.layers[layer_idx] = PrunedQwen2DecoderLayer(config, layer_idx)


class PrunedQwen2ForCausalLM(Qwen2ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, pruned_layers: List[int]):
        super().__init__(config)
        self.model = PrunedQwen2Model(config, pruned_layers)


def train(model, dataset):
    pass

def eval(model, dataset):
    pass

class SimpleDataLoader(Dataset):
    def __init__(self, dataset, tokenizer, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer

        self.tokens = []
        for item in self.dataset.select_columns("tokens"):
            self.tokens.append(self.tokenizer(item["tokens"], return_tensors="pt", padding=True, truncation=True), padding_side="left")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.tokens[idx]

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--pruned_state_dict", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="lighteval/summarization")
    parser.add_argument("--config", type=str, default="cnn-dm")

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    pruned_state_dict = torch.load(args.pruned_state_dict)

    # Figure out the pruned layers
    pruned_layers = [1, 22, 2]
    # for key, value in pruned_state_dict.items():
    #     if torch.isnan(value).any():
    #         pruned_layers.append(int(key.split(".")[1]))

    # pdb.set_trace()

    pruned_model = PrunedQwen2ForCausalLM.from_pretrained(
        args.model_path,
        pruned_layers=pruned_layers
    )

    train_dataset = load_dataset(args.dataset, args.config, split="train")
    eval_dataset = load_dataset(args.dataset, args.config, split="validation")

    train_dataloader = SimpleDataLoader(train_dataset, tokenizer, 16)
    eval_dataloader = SimpleDataLoader(eval_dataset, tokenizer, 16)

    train(pruned_model, train_dataloader)
    eval(pruned_model, eval_dataloader)

    # Save the model
    suffix = args.pruned_state_dict.split("/")[-1].split("_")[-1].split(".")[0]
    pruned_model.save_pretrained(f"models/pruned_model_state_dict_{suffix}.safetensors")
