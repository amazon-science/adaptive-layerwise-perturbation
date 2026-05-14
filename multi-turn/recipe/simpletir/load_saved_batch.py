#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Script to load and inspect saved batch data from training.

Usage:
    python load_saved_batch.py /path/to/saved_batch.pt
    
    Or if you want to specify the experiment directory:
    python load_saved_batch.py --exp_dir ${CHECKPOINT_PATH:-./checkpoints}/your_experiment_name
"""

import argparse
import os
from pprint import pprint

import torch
from transformers import AutoTokenizer


def print_tensor_info(name, tensor):
    """Print information about a tensor."""
    if isinstance(tensor, torch.Tensor):
        print(f"  {name}:")
        print(f"    - shape: {tensor.shape}")
        print(f"    - dtype: {tensor.dtype}")
        print(f"    - device: {tensor.device}")
        if tensor.numel() > 0:
            print(f"    - min: {tensor.min().item():.4f}")
            print(f"    - max: {tensor.max().item():.4f}")
            print(f"    - mean: {tensor.float().mean().item():.4f}")
        print()
    else:
        print(f"  {name}: {type(tensor)}")


def load_and_inspect_batch(file_path, tokenizer_path=None, verbose=False):
    """Load and inspect a saved batch file."""
    print(f"Loading batch data from: {file_path}")
    print("=" * 80)
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    # Load the data
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    
    print(f"\n📦 Batch Data Structure:")
    print(f"  - Keys: {list(data.keys())}")
    print()
    
    # Inspect batch tensors
    if 'batch_tensors' in data:
        batch_tensors = data['batch_tensors']
        print(f"🔢 Batch Tensors ({len(batch_tensors)} tensors):")
        print(f"  Keys: {list(batch_tensors.keys())}")
        print()
        
        for key, tensor in batch_tensors.items():
            print_tensor_info(key, tensor)
    
    # Inspect non-tensor batch
    if 'batch_non_tensors' in data:
        batch_non_tensors = data['batch_non_tensors']
        print(f"📝 Non-Tensor Batch ({len(batch_non_tensors)} items):")
        print(f"  Keys: {list(batch_non_tensors.keys())}")
        print()
        
        for key, value in batch_non_tensors.items():
            print(f"  {key}:")
            if isinstance(value, list):
                print(f"    - type: list")
                print(f"    - length: {len(value)}")
                if len(value) > 0:
                    print(f"    - first item type: {type(value[0])}")
                    if verbose and len(value) <= 5:
                        print(f"    - items: {value}")
            else:
                print(f"    - type: {type(value)}")
                if verbose:
                    print(f"    - value: {value}")
            print()
    
    # Inspect meta info
    if 'batch_meta_info' in data:
        batch_meta_info = data['batch_meta_info']
        print(f"ℹ️  Meta Info ({len(batch_meta_info)} items):")
        print(f"  Keys: {list(batch_meta_info.keys())}")
        print()
        
        for key, value in batch_meta_info.items():
            print(f"  {key}: {value}")
        print()
    
    # Decode some samples if tokenizer is provided
    if tokenizer_path and 'batch_tensors' in data and 'input_ids' in data['batch_tensors']:
        print("=" * 80)
        print(f"🔤 Decoding samples with tokenizer: {tokenizer_path}")
        print()
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            input_ids = data['batch_tensors']['input_ids']
            
            num_samples_to_show = min(3, input_ids.shape[0])
            for i in range(num_samples_to_show):
                print(f"Sample {i + 1}:")
                decoded = tokenizer.decode(input_ids[i], skip_special_tokens=False)
                print(f"{decoded[:500]}..." if len(decoded) > 500 else decoded)
                print()
            
            # Decode responses if available
            if 'responses' in data['batch_tensors']:
                responses = data['batch_tensors']['responses']
                print(f"Responses shape: {responses.shape}")
                for i in range(num_samples_to_show):
                    print(f"Response {i + 1}:")
                    decoded_response = tokenizer.decode(responses[i], skip_special_tokens=False)
                    print(f"{decoded_response[:500]}..." if len(decoded_response) > 500 else decoded_response)
                    print()
        
        except Exception as e:
            print(f"Error loading tokenizer or decoding: {e}")
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Load and inspect saved batch data from training"
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        type=str,
        help="Path to the saved batch file (.pt)",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        help="Experiment directory (will look for debug_data/first_step_batch.pt)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Tokenizer path for decoding text (default: Qwen/Qwen2.5-7B)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output including all non-tensor data",
    )
    
    args = parser.parse_args()
    
    # Determine file path
    if args.file_path:
        file_path = args.file_path
    elif args.exp_dir:
        file_path = os.path.join(args.exp_dir, "debug_data", "first_step_batch.pt")
    else:
        parser.error("Either file_path or --exp_dir must be provided")
    
    # Load and inspect
    load_and_inspect_batch(file_path, args.tokenizer, args.verbose)
    
    print("=" * 80)
    print("✅ Done!")


if __name__ == "__main__":
    main()

