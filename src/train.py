# Load the pruned model and infer

import pdb
import json
import torch

from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

from .utils import prepare_train_dataset
from .modeling import PrunedQwen2ForCausalLM

config = {
    "batch_size": 4,
    "per_device_eval_batch_size": 2,
    "eval_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "num_epochs": 2,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 1000,
    "torch_empty_cache_steps": 100,
}

def train(model, train_dataset, eval_dataset, tokenizer):
    print(f"Starting training...")

    sft_config = SFTConfig(
        max_length=1024,
        output_dir="./results",
        packing=False,
        do_train=True,
        do_eval=False,
        # eval_strategy=config["eval_strategy"],
        # eval_steps=config["eval_steps"],
        # num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        # per_device_eval_batch_size=config["per_device_eval_batch_size"],
        # eval_accumulation_steps=config["eval_accumulation_steps"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=config["gradient_checkpointing"],
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        warmup_ratio=config["warmup_ratio"],
        metric_for_best_model="train_loss",
        # load_best_model_at_end=True,
        torch_empty_cache_steps=config["torch_empty_cache_steps"],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        processing_class=tokenizer
    )

    trainer.train()
    eval_results = trainer.evaluate()

    print("\nFinal Evaluation Results:")
    print(f"Loss: {eval_results['eval_loss']:.4f}")
    
    return trainer


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--pruned_metrics", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="lighteval/summarization")
    parser.add_argument("--config", type=str, default="cnn-dm")

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    pruned_layers = json.load(open(args.pruned_metrics))["layers_to_prune"]
    print(f"Pruned layers: {pruned_layers}")

    pruned_model = PrunedQwen2ForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        pruned_layers=pruned_layers,
        use_cache=False,
    ).to("cuda")

    param_count = sum(p.numel() for p in pruned_model.parameters()) * 2
    print(f"Pruned model size: {param_count / 1e9} GB")

    train_dataset = prepare_train_dataset(args.dataset, args.config, split="train", tokenizer=tokenizer)
    eval_dataset = prepare_train_dataset(args.dataset, args.config, split="validation", tokenizer=tokenizer)
    eval_dataset = eval_dataset.shuffle().select(range(512))

    with torch.no_grad():
        print("Sample forward pass...")
        msg = [
            {
                "role": "system",
                "content": "You are an helpful AI assistant whose job is to provide a concise summarize of the given content"
            },
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ]
        inputs = tokenizer.apply_chat_template(msg, tokenize=False)
        inputs = tokenizer(inputs, return_tensors="pt").to("cuda")
        outputs = pruned_model.generate(**inputs, max_new_tokens=16, use_cache=False)

        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    train(pruned_model, train_dataset, eval_dataset, tokenizer)

    # Save the model
    suffix = args.pruned_state_dict.split("/")[-1].split("_")[-1].split(".")[0]
    pruned_model.save_pretrained(f"models/pruned_model_state_dict_{suffix}.safetensors")

    with torch.no_grad():
        print("Sample forward pass after training...")
        msg = [
            {
                "role": "system",
                "content": "You are an helpful AI assistant whose job is to provide a concise summarize of the given content"
            },
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ]
        inputs = tokenizer.apply_chat_template(msg, tokenize=False)
        inputs = tokenizer(inputs, return_tensors="pt").to("cuda")
        outputs = pruned_model.generate(**inputs, max_new_tokens=16, use_cache=False)

        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
