import os
import json
import torch
import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import track
from dotenv import load_dotenv

from .utils import load_model, prepare_dataset
from .modeling import PrunedQwen2ForCausalLM


console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Compare two models using an LLM judge")
    parser.add_argument("--original_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Path to the original model")
    parser.add_argument("--pruned_model", type=str, default="./models/pruned_model_state_dict_115217.safetensors", help="Path to the pruned model")
    parser.add_argument("--pruned_metrics", type=str, default="./models/pruned_metrics_20250406_025700.json", help="Path to the pruned metrics")
    parser.add_argument("--dataset_name", type=str, default="lighteval/summarization", help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="cnn-dm", help="Dataset configuration")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output_file", type=str, default="results/model_comparison.json", help="Output file for results")

    return parser.parse_args()

def generate_response(model, tokenizer, prompt, max_length=1024):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()

    return response

def judge_responses(question, response1, response2, llm_api_endpoint, llm_api_key):
    """Use the LLM API to judge which response is better."""
    import requests

    prompt = f"""
    You are an expert judge evaluating the quality of AI model responses.
    Please evaluate which response is better. Consider factors like:
    - Accuracy and correctness
    - Completeness
    - Clarity and coherence
    - Relevance to the question
    
    Respond with ONLY one of these options:
    - "MODEL1": If Model 1's response is clearly better
    - "MODEL2": If Model 2's response is clearly better
    - "TIE": If both responses are equally good or equally flawed

    ------------------------------------------------------------------------------------------------
    Question:
    {question}
    ------------------------------------------------------------------------------------------------
    Response from Model 1:
    {response1}
    ------------------------------------------------------------------------------------------------
    Response from Model 2:
    {response2}
    ------------------------------------------------------------------------------------------------
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {llm_api_key}"
    }
    
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(llm_api_endpoint, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        judgment = result["choices"][0]["message"]["content"].strip().upper()

        # Normalize the judgment
        if "MODEL1" in judgment:
            return "original_model"
        elif "MODEL2" in judgment:
            return "pruned_model"
        else:
            return "tie"
    except Exception as e:
        console.print(f"[red]Error calling LLM API: {e}[/red]")
        return "ERROR"

def evaluate_models(args):
    """Evaluate two models on a dataset and compare their outputs using an LLM judge."""

    llm_api_endpoint = os.getenv("llm_api_endpoint")
    llm_api_key = os.getenv("llm_api_key")

    console.print("[bold green]Loading models...[/bold green]")

    pruned_layers = json.load(open(args.pruned_metrics))["layers_to_prune"]
    print(f"Pruned layers: {pruned_layers}")

    original_tokenizer, original_model = load_model(args.original_model)
    pruned_model = PrunedQwen2ForCausalLM.from_pretrained(
        args.pruned_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
        pruned_layers=pruned_layers
    ).to("cuda")

    original_model_size = round(sum(p.numel() for p in original_model.parameters()) * 2 / 1e9, 2)
    pruned_model_size = round(sum(p.numel() for p in pruned_model.parameters()) * 2 / 1e9, 2)

    print(f"Oringal model size: {original_model_size} GB")
    print(f"Pruned model size: {pruned_model_size} GB")
    print(f"Saving in memory: {(original_model_size - pruned_model_size)/original_model_size * 100}%")

    original_model.eval()
    pruned_model.eval()

    console.print(f"[bold green]Loading dataset {args.dataset_name}...[/bold green]")
    dataset = prepare_dataset(args.dataset_name, args.dataset_config, args.dataset_split, original_tokenizer)

    # Limit dataset size
    if args.num_samples > 0:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    results = {
        "original_model": args.original_model,
        "pruned_model": args.pruned_model,
        "dataset": args.dataset_name,
        "num_samples": len(dataset),
        "comparisons": [],
        "summary": {
            "original_model": 0,
            "pruned_model": 0,
            "tie": 0,
            "error": 0
        }
    }
    
    console.print("[bold green]Evaluating models...[/bold green]")
    for i, sample in enumerate(track(dataset, description="Processing samples")):
    # for i, sample in enumerate(dataset):
        question = "Summarize the following article: " + sample["article"]

        response1 = generate_response(original_model, original_tokenizer, question)
        response2 = generate_response(pruned_model, original_tokenizer, question)

        judgment = judge_responses(question, response1, response2, llm_api_endpoint, llm_api_key)

        # Record result
        comparison = {
            "sample_id": i,
            "question": question,
            "response1": response1,
            "response2": response2,
            "judgment": judgment
        }
        results["comparisons"].append(comparison)

        results["summary"][judgment.lower()] += 1

    # Calculate percentages
    total = len(dataset)
    results["summary"]["original_model_pct"] = results["summary"]["original_model"] / total * 100
    results["summary"]["pruned_model_pct"] = results["summary"]["pruned_model"] / total * 100
    results["summary"]["tie_pct"] = results["summary"]["tie"] / total * 100
    results["summary"]["error_pct"] = results["summary"]["error"] / total * 100
    
    # Determine overall winner
    if results["summary"]["original_model"] > results["summary"]["pruned_model"]:
        results["overall_winner"] = "original_model"
    elif results["summary"]["pruned_model"] > results["summary"]["original_model"]:
        results["overall_winner"] = "pruned_model"
    else:
        results["overall_winner"] = "tie"
    
    # Save results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Display summary
    console.print("\n[bold green]Evaluation Summary:[/bold green]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Count")
    table.add_column("Percentage")
    
    table.add_row("Original Model Better", str(results["summary"]["original_model"]), f"{results['summary']['original_model_pct']:.2f}%")
    table.add_row("Pruned Model Better", str(results["summary"]["pruned_model"]), f"{results['summary']['pruned_model_pct']:.2f}%")
    table.add_row("Tie", str(results["summary"]["tie"]), f"{results['summary']['tie_pct']:.2f}%")
    table.add_row("Error", str(results["summary"]["error"]), f"{results['summary']['error_pct']:.2f}%")
    
    console.print(table)
    console.print(f"\n[bold green]Overall Winner: {results['overall_winner']}[/bold green]")
    console.print(f"[bold green]Results saved to {args.output_file}[/bold green]")

if __name__ == "__main__":
    args = parse_args()
    load_dotenv(".env")
    evaluate_models(args)
