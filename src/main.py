import pdb
import torch
from typing import Dict, List
from torch.cuda import empty_cache
import logging
import time
import json
from datetime import datetime
import numpy as np
from .utils import load_model, prepare_dataset
from .scores import compute_attention_entropy, compute_attention_kl, compute_composite_score


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
{layer1: [step 1, step 2, ...], layer2: [step 1, step 2, ...]}
"""
activation_scores: Dict[str, List[torch.Tensor]] = {}
attention_scores: Dict[str, List[torch.Tensor]] = {}

"""
{
    layer1: {head1: [batch 1, batch 2, ...], head2: [batch 1, batch 2, ...]},
}
"""
attention_entropies: Dict[str, List[float]] = {}
attention_kl: Dict[str, List[float]] = {}


def activation_hook(layer_name):
    def hook(module, input, output):
        if layer_name not in activation_scores:
            activation_scores[layer_name] = []
        activation_scores[layer_name].append(output.detach().cpu())
    return hook

def attention_hook(layer_name):
    def hook(module, input, output):
        _, attn_weights, _ = output
        if layer_name not in attention_scores:
            attention_scores[layer_name] = []
        attention_scores[layer_name].append(attn_weights.detach().cpu())
    return hook

def register_hooks(model, activation_handles, attention_handles):
    for i, block in enumerate(model.model.layers):
        if hasattr(block, "mlp"):
            activation_handles.append(block.mlp.register_forward_hook(activation_hook(f"layer_{i}_mlp")))
        if hasattr(block, "self_attn"):
            attention_handles.append(block.self_attn.register_forward_hook(attention_hook(f"layer_{i}_self_attn")))

def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", type=str, default="lighteval/summarization")
    parser.add_argument("--config", type=str, default="cnn-dm")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=3, help="Number of layers to prune")
    parser.add_argument("--visualization", type=str, choices=["cli", "plot"], default="cli", help="Visualization method: 'cli' for terminal-based or 'plot' for matplotlib")

    return parser.parse_args()

def compute_and_store_scores(decoding_steps):
    # Goes layer by layer, attention scores batch_size x num_heads x seq_len x seq_len
    for key, attn_list in attention_scores.items():
        entropies = []
        kls = []

        for tensor in attn_list:
            entropies.append(compute_attention_entropy(tensor))
            kls.append(compute_attention_kl(tensor))

        # Convert list of lists to tensor and average over decoding steps
        entropies_tensor = torch.tensor(entropies)  # shape: (num_steps, num_heads)
        kl_tensor = torch.tensor(kls)              # shape: (num_steps, num_heads)
        avg_entropy = entropies_tensor.mean(dim=0).tolist()
        avg_kl = kl_tensor.mean(dim=0).tolist()

        if key not in attention_entropies:
            attention_entropies[key] = {}
        if key not in attention_kl:
            attention_kl[key] = {}

        for h in range(len(avg_entropy)):
            if h not in attention_entropies[key]:
                attention_entropies[key][h] = []
            if h not in attention_kl[key]:
                attention_kl[key][h] = []

            attention_entropies[key][h].append(avg_entropy[h])
            attention_kl[key][h].append(avg_kl[h])

def plot_metrics():
    import matplotlib.pyplot as plt
    import numpy as np

    # Plot a bar for every head in every layer
    # layer are x-axis, head is y-axis
    # since heads are the same for every layer, can we just assign the same color for easy visualization?
    # Create a figure for each metric
    for metric in ["entropy", "kl"]:
        plt.figure(figsize=(12, 8))
        for layer in attention_entropies.keys():
            for head in attention_entropies[layer].keys():
                data = attention_entropies[layer][head] if metric == "entropy" else attention_kl[layer][head]
                plt.bar(layer, data, label=f'Head {head}', color=f'C{head}')

        plt.title(f'{metric.capitalize()} by Layer')
        plt.xticks(rotation=90)
        plt.xlabel('Layer')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.show()

def visualize_metrics_cli(idx):
    """Visualize metrics in the CLI using rich library"""
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    
    console = Console()
    
    # Create a layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body")
    )
    
    # Create tables for entropy and KL
    entropy_table = Table(title="Attention Entropy by Layer and Head")
    kl_table = Table(title="Attention KL by Layer and Head")
    
    # Get all unique layers and heads
    layers = attention_entropies.keys()
    heads = set(head for layer in attention_entropies.values() for head in layer.keys())
 
    # Add columns
    entropy_table.add_column("Layer", style="cyan")
    kl_table.add_column("Layer", style="cyan")
    
    for head in heads:
        entropy_table.add_column(f"Head {head}", style=f"color({head+1})")
        kl_table.add_column(f"Head {head}", style=f"color({head+1})")
    
    # Add rows
    for layer in layers:
        entropy_row = [layer]
        kl_row = [layer]
        
        for head in heads:
            if head in attention_entropies[layer]:
                entropy_row.append(f"{np.mean(attention_entropies[layer][head]):.4f}")
                kl_row.append(f"{np.mean(attention_kl[layer][head]):.4f}")
            else:
                entropy_row.append("N/A")
                kl_row.append("N/A")
        
        entropy_table.add_row(*entropy_row)
        kl_table.add_row(*kl_row)
    
    # Create panels for each table
    entropy_panel = Panel(entropy_table, title="Entropy Metrics", border_style="green")
    kl_panel = Panel(kl_table, title="KL Metrics", border_style="blue")

    # Update layout
    layout["header"].update(Panel(f"Attention Metrics Visualization Batch: {idx}", style="bold white"))
    layout["body"].update(entropy_panel)
    # layout["body"].update(kl_panel)
    
    # Display the layout
    console.print(layout)

def print_metrics():
    """
    Print layer wise average metrics in a table
    """
    from rich.table import Table
    from rich.console import Console

    console = Console()
    table = Table(title="Layer Wise Metrics")
    
    table.add_column("Layer")
    table.add_column("Entropy")
    table.add_column("KL")

    for layer in attention_entropies.keys():
        avg_entropy = sum(attention_entropies[layer].values()) / len(attention_entropies[layer])
        avg_kl = sum(attention_kl[layer].values()) / len(attention_kl[layer])
        table.add_row(layer, f"{avg_entropy:.4f}", f"{avg_kl:.4f}")

    console.print(table)

def main():
    global activation_scores
    global attention_scores

    args = parse_args()

    tokenizer, model = load_model(args.model)
    dataset = prepare_dataset(args.dataset, args.config, args.split, tokenizer)

    logger.info(f"Loaded model and dataset")

    activation_handles = []
    attention_handles = []
    register_hooks(model, activation_handles, attention_handles)

    tmp = dataset.select_columns("tokens").to_dict()
    input_prompts = [v for v in tmp.values()][0]

    # for bs in tqdm(range(0, len(input_prompts), args.batch_size), desc="Processing batches"):
    for idx, bs in enumerate(range(0, len(input_prompts), args.batch_size)):
        activation_scores = {}
        attention_scores = {}

        be = bs+args.batch_size
        input_tokens = tokenizer(input_prompts[bs:be], return_tensors="pt", truncation=True, padding=True, padding_side="left")
        input_tokens = {k: v.cuda() for k, v in input_tokens.items()}

        # After every forward pass compute the scores
        # Clear the activations and attentions for the next batch

        with torch.no_grad():
            outputs = model.generate(
                **input_tokens,
                return_dict_in_generate=True,
                output_attentions=True,
                output_hidden_states=True,
                max_length=1024
            )['sequences'].cpu()

        # find the number of decoding steps for each sample
        decoding_steps = []
        for inp, out in zip(input_tokens["input_ids"], outputs):
            out = out[len(inp):]
            decoding_steps.append(torch.where(out == 151645)[0][0])

        compute_and_store_scores(decoding_steps)
        
        # Update visualization if using CLI
        if args.visualization == "cli":
            visualize_metrics_cli(idx)
            time.sleep(0.5)  # Small delay to make the visualization readable
            
        empty_cache()

    if args.visualization == "plot":
        plot_metrics()
    else:
        visualize_metrics_cli(idx)

    # Average over different batches
    for layer in attention_entropies.keys():
        for head in attention_entropies[layer].keys():
            attention_entropies[layer][head] = np.mean(attention_entropies[layer][head])
            attention_kl[layer][head] = np.mean(attention_kl[layer][head])

    print_metrics()

    layers_to_prune = compute_composite_score(attention_entropies, attention_kl, top_k=args.top_k)

    logger.info(f"Layers to prune: {layers_to_prune}")

    suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info("Pruning the model and exporting the pruned model")

    # Export the torch state dict of the model
    # but NaN the weights of the heads that in the composite scores
    model_state_dict = model.state_dict()

    for layer, val in model_state_dict.items():
        if "self_attn" in layer and layer.split(".")[2] in layers_to_prune:
            model_state_dict[layer].copy_(torch.ones_like(model_state_dict[layer]) * float('nan'))
            logger.info(f"NaN'd layer {layer}")

    torch.save(model_state_dict, f"models/model_state_dict_{suffix}.pth")

    logger.info("Exporting the metrics")
    with open(f"models/metrics_{suffix}.json", "w") as f:
        json.dump({
            "attention_entropies": attention_entropies,
            "attention_kl": attention_kl,
            "layers_to_prune": layers_to_prune
        }, f)

if __name__ == "__main__":
    main()
