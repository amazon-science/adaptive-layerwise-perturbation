#!/usr/bin/env python3
"""
Download HuggingFace datasets and save them as parquet files
"""
import os
import argparse
from pathlib import Path
from datasets import load_dataset


def download_and_save_dataset(dataset_name, output_dir, split="train"):
    """
    Download a dataset from HuggingFace and save as parquet
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., 'weqweasdas/math500')
        output_dir: Output directory to save parquet files
        split: Dataset split to download (default: 'test')
    """
    print(f"Downloading dataset: {dataset_name}")
    
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset(dataset_name, split=split)
        
        # Rename 'problem' column to 'prompt' if it exists
        if 'problem' in dataset.column_names:
            print(f"  Renaming 'problem' column to 'prompt'")
            dataset = dataset.rename_column('problem', 'prompt')
        
        # Convert prompt to chat format [{'content': '...', 'role': 'user'}]
        if 'prompt' in dataset.column_names:
            print(f"  Converting 'prompt' to chat format")
            def convert_prompt_to_chat(example):
                if isinstance(example['prompt'], str):
                    example['prompt'] = [{'content': example['prompt'], 'role': 'user'}]
                elif isinstance(example['prompt'], list) and len(example['prompt']) > 0:
                    # Already in list format, check if needs role/content keys
                    if isinstance(example['prompt'][0], str):
                        example['prompt'] = [{'content': example['prompt'][0], 'role': 'user'}]
                return example
            
            dataset = dataset.map(convert_prompt_to_chat)
        
        # Add data_source column if not exists
        if 'data_source' not in dataset.column_names:
            print(f"  Adding 'data_source' column")
            dataset_source_name = dataset_name.split('/')[-1] if '/' in dataset_name else dataset_name
            dataset = dataset.add_column('data_source', [dataset_source_name] * len(dataset))
        
        # Add ability column if not exists (default to 'math')
        if 'ability' not in dataset.column_names:
            print(f"  Adding 'ability' column")
            dataset = dataset.add_column('ability', ['math'] * len(dataset))
        
        # Add extra_info column if not exists
        if 'extra_info' not in dataset.column_names:
            print(f"  Adding 'extra_info' column")
            def add_extra_info(example, idx):
                example['extra_info'] = {'index': idx, 'split': split}
                return example
            
            dataset = dataset.map(add_extra_info, with_indices=True)
        
        # Convert 'gt' to 'reward_model' format (like deepscaler)
        if 'gt' in dataset.column_names:
            print(f"  Converting 'gt' to 'reward_model' format")
            def convert_gt_to_reward_model(example):
                example['reward_model'] = {
                    'ground_truth': str(example['gt']),
                    'style': 'rule'
                }
                return example
            
            dataset = dataset.map(convert_gt_to_reward_model)
            # Remove the old 'gt' column
            dataset = dataset.remove_columns(['gt'])
            print(f"  Removed 'gt' column, added 'reward_model' column")
        
        # Create output directory matching the dataset name structure
        # e.g., weqweasdas/math500 -> output_dir/weqweasdas/
        if "/" in dataset_name:
            org, name = dataset_name.split("/")
            dataset_output_dir = Path(output_dir) / 'deepscaler'
        else:
            dataset_output_dir = Path(output_dir)
            name = dataset_name
        
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet
        output_path = dataset_output_dir / f"{name.split('/')[-1]}.parquet"
        dataset.to_parquet(str(output_path))
        
        print(f"✓ Saved to: {output_path}")
        print(f"  Number of samples: {len(dataset)}")
        print(f"  Columns: {dataset.column_names}")
        
        return output_path
        
    except Exception as e:
        print(f"✗ Error downloading {dataset_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets to local parquet files")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of HuggingFace dataset names"
    )
    parser.add_argument(
        "--output_dir",
        default="./datasets",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download (default: test)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Downloading HuggingFace Datasets")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Split: {args.split}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print()
    
    success_count = 0
    failed_datasets = []
    
    for dataset_name in args.datasets:
        result = download_and_save_dataset(
            dataset_name,
            args.output_dir,
            split=args.split
        )
        if result:
            success_count += 1
        else:
            failed_datasets.append(dataset_name)
        print()
    
    print("=" * 60)
    print(f"Download complete: {success_count}/{len(args.datasets)} successful")
    if failed_datasets:
        print(f"Failed datasets: {', '.join(failed_datasets)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

