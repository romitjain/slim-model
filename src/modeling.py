# Load the pruned model and infer

import pdb
import torch
from typing import List, Optional, Tuple

from transformers import AutoTokenizer, TrainingArguments
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from trl import SFTTrainer

from .utils import prepare_train_dataset

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


def train(model, train_dataset, eval_dataset, tokenizer):
    training_args = TrainingArguments(
        output_dir="./results",
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=100,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        logging_steps=10,
        warmup_ratio=0.1,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    train_results = trainer.train()
    eval_results = trainer.evaluate()
    
    print("\nFinal Evaluation Results:")
    print(f"Loss: {eval_results['eval_loss']:.4f}")
    
    return trainer


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
        pruned_layers=pruned_layers,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    train_dataset = prepare_train_dataset(args.dataset, args.config, split="train")
    eval_dataset = prepare_train_dataset(args.dataset, args.config, split="validation")

    train(pruned_model, train_dataset, eval_dataset, tokenizer)

    # Save the model
    suffix = args.pruned_state_dict.split("/")[-1].split("_")[-1].split(".")[0]
    pruned_model.save_pretrained(f"models/pruned_model_state_dict_{suffix}.safetensors")
