import json
import ast
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse

def parse_list_string(list_string: str) -> List[List[int]]:
    """
    Convert string representation of list of lists to actual nested list.
    Also handles cases where the list is embedded in a conversation.
    """
    try:
        # First try direct parsing
        return ast.literal_eval(list_string)
    except:
        try:
            # If direct parsing fails, try to extract the last list-like pattern
            # Look for the last occurrence of square brackets
            last_open_bracket = list_string.rfind('[')
            last_close_bracket = list_string.rfind(']')
            
            if last_open_bracket != -1 and last_close_bracket != -1:
                # Extract the substring that looks like a list
                potential_list = list_string[last_open_bracket:last_close_bracket + 1]
                
                # Try to parse the extracted list
                parsed_list = ast.literal_eval(potential_list)
                
                # Check if we need to wrap it in another list
                if isinstance(parsed_list, list):
                    # If the first element is an integer, wrap the whole list
                    if parsed_list and isinstance(parsed_list[0], int):
                        return [parsed_list]
                    # If the first element is a list, return as is
                    elif parsed_list and isinstance(parsed_list[0], list):
                        return parsed_list
                    
                # If we can't determine the structure, return empty list
                print(f"Unexpected list structure in: {potential_list}")
                return []
            
            print(f"Failed to find list pattern in string: {list_string}")
            return []
        except:
            print(f"Failed to parse list string: {list_string}")
            return []

def calculate_relaxed_match(pred_lists: List[List[int]], gt_lists: List[List[int]]) -> float:
    """
    Calculate relaxed matching score as a product of precision scores for each sublist.
    
    For each predicted sublist and corresponding ground truth sublist:
    - Calculates precision as (number of predicted elements in ground truth) / (number of predicted elements)
    - Returns product of precision scores across all sublists
    
    Returns:
    - 0.0 if number of sublists don't match
    - Product of precision scores (between 0.0 and 1.0) otherwise
    """
    # Check if number of sublists match
    if len(pred_lists) != len(gt_lists):
        return 0.0
    
    # Check each corresponding sublist pair
    precision_all_goals = []
    for pred_sublist, gt_sublist in zip(pred_lists, gt_lists):
        # If none of the predicted elements appear in ground truth sublist, return 0
        if len(pred_sublist) == 0 and len(gt_sublist) == 0:
            precision = 1.0
        elif len(pred_sublist) == 0 or len(gt_sublist) == 0:
            precision = 0.0
        else:
            precision = sum(pred_elem in gt_sublist for pred_elem in pred_sublist) / len(pred_sublist)
            precision_all_goals.append(precision)

    # multiply precision of all goals
    return np.prod(precision_all_goals)

def evaluate_results(results_file: str) -> Dict:
    """Evaluate both exact and relaxed matching metrics from results file."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    total_examples = len(data["individual_results"])
    exact_matches = 0
    relaxed_scores = []  # Changed from counter to list of scores
    
    for result in data["individual_results"]:
        try:
            # Get ground truth and model output
            gt_string = result["ground_truth"]
            pred_string = result["model_output"]
            
            # Calculate exact match
            if gt_string == pred_string:
                exact_matches += 1
            
            # Calculate relaxed match
            gt_lists = parse_list_string(gt_string)
            pred_lists = parse_list_string(pred_string)
            
            # If either parsing returned empty list (parsing failure), treat as 0 for relaxed metric
            if not gt_lists or not pred_lists:
                relaxed_scores.append(0.0)
                continue
                
            # Store the actual precision score
            relaxed_score = calculate_relaxed_match(pred_lists, gt_lists)
            relaxed_scores.append(relaxed_score)
                
        except Exception as e:
            # Treat exceptions as 0 for relaxed metric
            relaxed_scores.append(0.0)
            print(f"Error processing example: {e}")
            continue
    
    # Calculate accuracies
    exact_accuracy = exact_matches / total_examples if total_examples > 0 else 0
    avg_relaxed_score = np.mean(relaxed_scores) if relaxed_scores else 0
    
    return {
        "total_examples": total_examples,
        "exact_matches": exact_matches,
        "avg_relaxed_score": avg_relaxed_score,
        "exact_accuracy": exact_accuracy
    }

def evaluate_directory(root_dir: str, output_file: str):
    """
    Recursively evaluate all JSON files in nested directories and save combined results.
    Handles structure: root_dir/model_dir/evals/checkpoint-X.json
    """
    root_path = Path(root_dir)
    all_results = []
    
    # Find all model directories
    model_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    print(f"Found {len(model_dirs)} model directories to evaluate")
    
    # Process each model directory
    for model_dir in model_dirs:
        model_name = model_dir.name
        # Clean model name for display (remove "resume-" prefix)
        display_name = model_name
        if model_name.startswith("resume-"):
            display_name = model_name[len("resume-"):]
            
        evals_dir = model_dir / "evals"
        
        if not evals_dir.exists() or not evals_dir.is_dir():
            print(f"No 'evals' directory found in {model_name}, skipping")
            continue
            
        # Get all JSON files in the evals directory and skip "checkpoint-latest.json"
        json_files = [f for f in evals_dir.glob('*.json') if f.name != "checkpoint-latest.json"]
        print(f"Found {len(json_files)} JSON files in {model_name}/evals")
        
        # Evaluate each checkpoint file
        for json_file in json_files:
            print(f"\nEvaluating {model_name}/evals/{json_file.name}...")
            metrics = evaluate_results(str(json_file))
            
            # Extract checkpoint number
            checkpoint_name = json_file.stem  # Remove .json extension
            checkpoint_num = 0
            if checkpoint_name.startswith("checkpoint-"):
                try:
                    checkpoint_num = int(checkpoint_name.split("-")[1])
                except (ValueError, IndexError):
                    pass
            
            # Add model and checkpoint info to metrics
            metrics["model_name"] = model_name
            metrics["display_name"] = display_name  # Store cleaned name
            metrics["checkpoint_name"] = checkpoint_name
            metrics["checkpoint_num"] = checkpoint_num
            metrics["file_path"] = str(json_file)
            all_results.append(metrics)
            
            # Print individual file results
            print(f"Results for {model_name}/{checkpoint_name}:")
            print(f"Total examples: {metrics['total_examples']}")
            print(f"Exact matches: {metrics['exact_matches']}")
            print(f"Average relaxed score: {metrics['avg_relaxed_score']:.4f}")
            print(f"Exact accuracy: {metrics['exact_accuracy']:.4f}")
    
    # Create plots for the results
    if not all_results:
        print("No results to plot.")
        return
    
    # Group results by model using the display name
    model_results = {}
    for result in all_results:
        display_name = result["display_name"]
        if display_name not in model_results:
            model_results[display_name] = []
        model_results[display_name].append(result)
    
    # Sort results by checkpoint number within each model
    for model_name in model_results:
        model_results[model_name].sort(key=lambda x: x["checkpoint_num"])
    
    # Generate a unique visual style for each model
    # Define a set of high-contrast colors arranged for maximum sequential contrast
    colors = [
        '#FF0000',  # Red
        '#0000FF',  # Blue
        '#00FF00',  # Green
        '#800080',  # Purple
        '#FFFF00',  # Yellow
        '#00FFFF',  # Cyan
        '#FF00FF',  # Magenta
        '#000000',  # Black
        '#FFA500',  # Orange
        '#008000',  # Dark Green
        '#00CED1',  # Turquoise
        '#800000',  # Dark Red
        '#32CD32',  # LimeGreen
        '#000080',  # Navy
        '#FF4500',  # OrangeRed
        '#1E90FF',  # DodgerBlue
        '#8A2BE2',  # Violet
        '#696969',  # DimGray
        '#FF1493',  # DeepPink
        '#FFC0CB',  # Pink
    ]
    
    # Use varied line styles for additional differentiation
    line_styles = ['-', '--', '-.', ':']
    line_widths = [2.0, 2.5, 3.0]
    
    # Create a systematic combination that maximizes contrast
    style_combinations = []
    
    # Create combinations in order, ensuring adjacent combinations have maximum contrast
    for line_style in line_styles:
        for line_width in line_widths:
            for color in colors:
                style_combinations.append({
                    'color': color,
                    'linestyle': line_style,
                    'linewidth': line_width
                })
    
    # Assign a unique style to each model
    model_styles = {}
    model_names = list(model_results.keys())
    
    # Assign styles sequentially - the style_combinations are already optimized for contrast
    for i, model_name in enumerate(model_names):
        style_idx = i % len(style_combinations)
        model_styles[model_name] = style_combinations[style_idx]
    
    # Create figure for exact accuracy
    plt.figure(figsize=(14, 10))
        
    for model_name, results in model_results.items():
        checkpoint_nums = [r["checkpoint_num"] for r in results]
        exact_accuracies = [r["exact_accuracy"] for r in results]
        
        style = model_styles[model_name]
        
        plt.plot(checkpoint_nums, exact_accuracies, 
                 color=style['color'],
                 linestyle=style['linestyle'],
                 linewidth=style['linewidth'],
                 marker='o',
                 markersize=3,
                 label=model_name)
    
    # Add vertical lines for epoch boundaries
    epoch_checkpoints = [2568, 5136, 7704, 10272, 12840]
    for i, checkpoint in enumerate(epoch_checkpoints):
        plt.axvline(x=checkpoint, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        # Use axes coordinates to position labels
        plt.annotate(f'Epoch {i+1}', xy=(checkpoint, 0.01), xycoords=('data', 'axes fraction'),
                 fontsize=12, rotation=90, va='bottom', ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="none", alpha=0.7))
    
    plt.xlabel('Checkpoint Number', fontsize=14)
    plt.ylabel('Exact Accuracy', fontsize=14)
    plt.title('Exact Accuracy by Checkpoint', fontsize=16, fontweight='bold')
    
    # Create a more readable legend with columns if there are many models
    legend_cols = 1
    if len(model_results) > 10:
        legend_cols = 2
    if len(model_results) > 20:
        legend_cols = 3
        
    plt.legend(fontsize=11, loc='best', framealpha=0.9, ncol=legend_cols)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    exact_plot_path = os.path.join(os.path.dirname(output_file), 'exact_accuracy_plot.png')
    plt.savefig(exact_plot_path, dpi=300, bbox_inches='tight')
    print(f"Exact accuracy plot saved to: {exact_plot_path}")
    plt.close()
    
    # Create figure for relaxed accuracy
    plt.figure(figsize=(14, 10))
    
    for model_name, results in model_results.items():
        checkpoint_nums = [r["checkpoint_num"] for r in results]
        relaxed_accuracies = [r["avg_relaxed_score"] for r in results]
        
        # Use exactly the same style as in the exact accuracy plot
        style = model_styles[model_name]
        
        plt.plot(checkpoint_nums, relaxed_accuracies, 
                 color=style['color'],
                 linestyle=style['linestyle'],
                 linewidth=style['linewidth'],
                 marker='o',
                 markersize=3,
                 label=model_name)
    
    # Add vertical lines for epoch boundaries
    for i, checkpoint in enumerate(epoch_checkpoints):
        plt.axvline(x=checkpoint, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        # Use axes coordinates to position labels
        plt.annotate(f'Epoch {i+1}', xy=(checkpoint, 0.01), xycoords=('data', 'axes fraction'),
                 fontsize=12, rotation=90, va='bottom', ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="none", alpha=0.7))
    
    plt.xlabel('Checkpoint Number', fontsize=14)
    plt.ylabel('Relaxed Accuracy', fontsize=14)
    plt.title('Relaxed Accuracy by Checkpoint', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9, ncol=legend_cols)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    relaxed_plot_path = os.path.join(os.path.dirname(output_file), 'relaxed_accuracy_plot.png')
    plt.savefig(relaxed_plot_path, dpi=300, bbox_inches='tight')
    print(f"Relaxed accuracy plot saved to: {relaxed_plot_path}")
    plt.close()
    
    # Calculate average metrics across all results
    num_files = len(all_results)
    avg_metrics = {
        "average_exact_accuracy": sum(r["exact_accuracy"] for r in all_results) / num_files if num_files > 0 else 0.0,
        "average_relaxed_score": sum(r["avg_relaxed_score"] for r in all_results) / num_files if num_files > 0 else 0.0,
        "total_files_evaluated": num_files,
        "total_examples_across_files": sum(r["total_examples"] for r in all_results),
    }
    
    # Combine all results
    final_results = {
        "individual_file_results": all_results,
        "average_metrics": avg_metrics,
        "model_names": list(model_results.keys())
    }
    
    # Save combined results
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nFinal Results:")
    print(f"Total models evaluated: {len(model_results)}")
    print(f"Total files evaluated: {avg_metrics['total_files_evaluated']}")
    print(f"Total examples across all files: {avg_metrics['total_examples_across_files']}")
    print(f"Average exact accuracy: {avg_metrics['average_exact_accuracy']:.4f}")
    print(f"Average relaxed score: {avg_metrics['average_relaxed_score']:.4f}")
    print(f"\nDetailed results saved to: {output_file}")

def main():
    """
    Main function to aggregate VLM checkpoint performance from multiple experiment runs.
    
    This script processes a directory structure containing multiple experiment runs,
    where each experiment subfolder contains training checkpoints and an 'evals' folder
    with JSON files containing evaluation results. It aggregates results into a single
    JSON file for exact/relaxed accuracy metrics and generates corresponding plots.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate and aggregate LLM outputs from multiple experiment runs"
    )
    
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing subfolders from various experiment runs. "
             "Each experiment subfolder should contain training checkpoints and an 'evals' "
             "folder with JSON files containing evaluation results of the checkpoints."
    )
    
    parser.add_argument(
        "--output_file", 
        type=str,
        required=True,
        help="Path to the output JSON file where aggregated results will be saved. "
             "This file will contain exact and relaxed accuracy metrics from all experiments."
    )
    
    args = parser.parse_args()
    
    evaluate_directory(args.root_dir, args.output_file)

if __name__ == "__main__":
    main()