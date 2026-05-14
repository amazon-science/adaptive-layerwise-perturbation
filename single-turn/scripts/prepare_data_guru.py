import os
import shutil
import pandas as pd

from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "LLM360/guru-RL-92k"
REPO_TYPE = "dataset"
LOCAL_DATA_DIR = os.path.expanduser(os.getenv("GURU_LOCAL_DATA_DIR", "~/data/guru_rl92k"))

all_files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
split_to_local = {"train": "train", "online_eval": "online_eval", "offline_eval": "offline_eval"}

def add_system_prompt(prompt_list):
    """Transform prompt from single user message to system + user messages.
    
    Args:
        prompt_list: List of message dicts, e.g. [{'content': '...', 'role': 'user'}]
    
    Returns:
        New list with system and user messages
    """
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    user_prompt_suffix = "Let's think step by step and output the final answer within \\boxed{}."
    old_prompt_suffix = "Please output the final answer within \\boxed{}."
    
    if not prompt_list or len(prompt_list) == 0:
        return prompt_list
    
    # Get original content
    original_content = prompt_list[0]['content']
    
    # Remove old suffix and add new one
    if old_prompt_suffix in original_content:
        problem = original_content.replace(old_prompt_suffix, "").strip()
    else:
        problem = original_content.strip()
    
    # Return new format with system + user messages
    return [
        {'content': system_prompt, 'role': 'system'},
        {'content': f"{problem} {user_prompt_suffix}", 'role': 'user'}
    ]


def normalize_difficulty_column(parquet_path):
    """Normalize difficulty column to int64, handling NaN values."""
    try:
        # Try reading with pyarrow first
        try:
            df = pd.read_parquet(parquet_path, engine='pyarrow')
        except Exception as e1:
            # If pyarrow fails due to nested data, try with different options
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(parquet_path, use_pandas_metadata=True)
                df = table.to_pandas()
            except Exception as e2:
                # Last resort: try fastparquet
                try:
                    df = pd.read_parquet(parquet_path, engine='fastparquet')
                except Exception as e3:
                    print(f"  ERROR: Failed to read {os.path.basename(parquet_path)}")
                    print(f"    pyarrow error: {str(e1)[:100]}")
                    print(f"    pyarrow table error: {str(e2)[:100]}")
                    print(f"    fastparquet error: {str(e3)[:100]}")
                    return False
    except Exception as e:
        print(f"  ERROR: Failed to read {os.path.basename(parquet_path)}: {e}")
        return False
    
    # 1. Normalize difficulty column if it exists
    if "difficulty" in df.columns:
        if df["difficulty"].dtype == "float64":
            mask = pd.notna(df["difficulty"])
            df.loc[mask, "difficulty"] = df.loc[mask, "difficulty"].round().astype("int64")
            df["difficulty"] = df["difficulty"].where(mask, pd.NA).astype("Int64")
            print(f"  Converted difficulty from float64 to Int64 in {os.path.basename(parquet_path)}")
        elif df["difficulty"].dtype == "int64":
            df["difficulty"] = df["difficulty"].astype("Int64")
            print(f"  Converted difficulty to nullable Int64 in {os.path.basename(parquet_path)}")
    
    # 2. Transform prompt format if prompt column exists
    if "prompt" in df.columns:
        # Check if already transformed (has system message)
        first_prompt = df['prompt'].iloc[0]
        if isinstance(first_prompt, list):
            prompt_list = first_prompt
        else:
            try:
                prompt_list = list(first_prompt)
            except:
                prompt_list = [first_prompt]
        
        # Skip if already has system message
        if len(prompt_list) > 1 or (len(prompt_list) == 1 and prompt_list[0].get('role') == 'system'):
            print(f"  Skipping prompt transformation (already transformed) in {os.path.basename(parquet_path)}")
        else:
            print(f"  Transforming prompt format in {os.path.basename(parquet_path)}...")
            new_prompts = []
            for i in range(len(df)):
                old_prompt = df['prompt'].iloc[i]
                # Convert to list if it's not already
                if isinstance(old_prompt, list):
                    prompt_list = old_prompt
                else:
                    # Handle numpy array or other types
                    try:
                        prompt_list = list(old_prompt)
                    except:
                        prompt_list = [old_prompt]
                
                # Transform the prompt
                new_prompt = add_system_prompt(prompt_list)
                new_prompts.append(new_prompt)
            
            df['prompt'] = new_prompts
            print(f"  Transformed {len(new_prompts)} prompts")
    
    # 3. Save back to parquet
    try:
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        print(f"  Saved normalized file: {os.path.basename(parquet_path)}")
        return True
    except Exception as e:
        print(f"  ERROR: Failed to save {os.path.basename(parquet_path)}: {e}")
        return False


def download_files_from_split(split, local_dir):
    parquet_files = [f for f in all_files if f.startswith(f"{split}/") and f.endswith(".parquet")]
    print(f"Downloading {len(parquet_files)} files to {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    for filename in parquet_files:
        print(f"Downloading {filename} to {local_dir}")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        # Remove .cache under each split folder
        cache_dir = os.path.join(local_dir, ".cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        
        # Normalize all downloaded files
        local_file_path = os.path.join(local_dir, filename)
        if os.path.exists(local_file_path):
            normalize_difficulty_column(local_file_path)


if __name__ == "__main__":
    # Option to normalize existing files
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--normalize-existing":
        # Normalize existing files in all splits
        for split in ["train", "online_eval", "offline_eval"]:
            split_dir = os.path.join(LOCAL_DATA_DIR, split)
            if os.path.exists(split_dir):
                print(f"\n{'='*60}")
                print(f"Normalizing existing {split} files...")
                print('='*60)
                success_count = 0
                error_count = 0
                for filename in sorted(os.listdir(split_dir)):
                    if filename.endswith(".parquet"):
                        file_path = os.path.join(split_dir, filename)
                        print(f"\nProcessing {filename}...")
                        if normalize_difficulty_column(file_path):
                            success_count += 1
                        else:
                            error_count += 1
                print(f"\nCompleted {split}: {success_count} files processed, {error_count} files failed")
            else:
                print(f"Directory {split_dir} does not exist")
    else:
        # Download and normalize files
        for split, local_dir in split_to_local.items():
            # download_files_from_split(split, os.path.join(LOCAL_DATA_DIR, local_dir))
            download_files_from_split(split, LOCAL_DATA_DIR)
