#!/usr/bin/env python3
"""Merge OpenR1 and Guru-RL92k datasets with de-duplication by user prompt."""

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def extract_user_content(prompt):
    """Extract the user message content from a prompt list."""
    if isinstance(prompt, (list, np.ndarray)):
        for item in prompt:
            if isinstance(item, dict) and item.get("role") == "user":
                return item.get("content", "")
    return ""


def merge_test_datasets(openr1_test_path: str, guru_test_paths: List[str], output_path: str, keep: str = "first"):
    """Merge evaluation sets and de-duplicate by extracted user content."""
    print("=" * 60)
    print("Test set merge + dedup utility")
    print("=" * 60)

    print("\n[1/5] Reading test datasets...")
    df_openr1 = pd.read_parquet(openr1_test_path)
    print(f"  - openr1 test: {df_openr1.shape[0]} rows, {df_openr1.shape[1]} cols")

    df_guru_list = []
    for i, path in enumerate(guru_test_paths):
        df = pd.read_parquet(path)
        df_guru_list.append(df)
        print(f"  - guru test[{i + 1}]: {df.shape[0]} rows, {df.shape[1]} cols")

    df_guru = pd.concat(df_guru_list, ignore_index=True) if len(df_guru_list) > 1 else df_guru_list[0]
    print(f"  - guru_rl92k total: {df_guru.shape[0]} rows")

    print("\n[2/5] Extracting user content for dedup...")
    df_openr1["user_content"] = df_openr1["prompt"].apply(extract_user_content)
    df_guru["user_content"] = df_guru["prompt"].apply(extract_user_content)

    df_openr1["_source_dataset"] = "openr1"
    df_guru["_source_dataset"] = "guru_rl92k"

    print("\n[3/5] Aligning schema...")
    all_cols_set = set(df_openr1.columns) | set(df_guru.columns)
    all_cols_set.discard("user_content")
    all_cols_set.discard("_source_dataset")

    for col in all_cols_set:
        if col not in df_openr1.columns:
            df_openr1[col] = None
        if col not in df_guru.columns:
            df_guru[col] = None

    priority_cols = sorted(all_cols_set)
    final_cols = priority_cols + ["user_content", "_source_dataset"]
    df_openr1 = df_openr1[final_cols]
    df_guru = df_guru[final_cols]

    print("\n[4/5] Merging datasets...")
    if keep == "guru":
        df_merged = df_guru.copy()
        openr1_unique = df_openr1[~df_openr1["user_content"].isin(df_guru["user_content"])]
        df_merged = pd.concat([df_merged, openr1_unique], ignore_index=True)
        print("  Strategy: keep guru first, then append unique openr1 samples")
    elif keep == "openr1":
        df_merged = df_openr1.copy()
        guru_unique = df_guru[~df_guru["user_content"].isin(df_openr1["user_content"])]
        df_merged = pd.concat([df_merged, guru_unique], ignore_index=True)
        print("  Strategy: keep openr1 first, then append unique guru samples")
    else:
        df_merged = pd.concat([df_openr1, df_guru], ignore_index=True)
        df_merged = df_merged.drop_duplicates(subset=["user_content"], keep=keep)
        print(f"  Strategy: pandas drop_duplicates(keep='{keep}')")

    print(f"  - merged: {df_merged.shape[0]} rows, {df_merged.shape[1]} cols")

    print("\n[5/5] Summary")
    print(f"  - raw total: {df_openr1.shape[0] + df_guru.shape[0]}")
    print(f"  - deduplicated: {df_merged.shape[0]}")
    print(f"  - removed duplicates: {df_openr1.shape[0] + df_guru.shape[0] - df_merged.shape[0]}")

    source_counts = df_merged["_source_dataset"].value_counts()
    print("  - source distribution:")
    for source, count in source_counts.items():
        print(f"    {source}: {count} ({count / df_merged.shape[0] * 100:.2f}%)")

    print(f"\nSaving to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_merged_final = df_merged.drop(columns=["user_content"])
    df_merged_final.to_parquet(output_path, index=False)
    print("Saved.")

    return df_merged_final


def merge_and_deduplicate(openr1_path: str, guru_path: str, output_path: str, keep: str = "first"):
    """Merge training sets and de-duplicate by extracted user content."""
    print("=" * 60)
    print("Train set merge + dedup utility")
    print("=" * 60)

    print("\n[1/5] Reading datasets...")
    df_openr1 = pd.read_parquet(openr1_path)
    df_guru = pd.read_parquet(guru_path)

    print(f"  - openr1: {df_openr1.shape[0]} rows, {df_openr1.shape[1]} cols")
    print(f"  - guru_rl92k: {df_guru.shape[0]} rows, {df_guru.shape[1]} cols")

    print("\n[2/5] Extracting user content for dedup...")
    df_openr1["user_content"] = df_openr1["prompt"].apply(extract_user_content)
    df_guru["user_content"] = df_guru["prompt"].apply(extract_user_content)

    df_openr1["_source_dataset"] = "openr1"
    df_guru["_source_dataset"] = "guru_rl92k"

    print("\n[3/5] Aligning schema...")
    guru_only_cols = [
        "source",
        "domain",
        "llama8b_solve_rate",
        "apply_chat_template",
        "is_unique",
        "solution",
        "qwen2.5_7b_pass_rate",
        "qwen3_30b_pass_rate",
    ]

    for col in guru_only_cols:
        if col not in df_openr1.columns:
            df_openr1[col] = None

    all_cols = list(df_guru.columns) + [col for col in df_openr1.columns if col not in df_guru.columns]
    priority_cols = [c for c in all_cols if c not in ["user_content", "_source_dataset"]]
    final_cols = priority_cols + ["user_content", "_source_dataset"]

    df_openr1 = df_openr1[[c for c in final_cols if c in df_openr1.columns]]
    df_guru = df_guru[[c for c in final_cols if c in df_guru.columns]]

    print("\n[4/5] Merging datasets...")
    if keep == "guru":
        df_merged = df_guru.copy()
        openr1_unique = df_openr1[~df_openr1["user_content"].isin(df_guru["user_content"])]
        df_merged = pd.concat([df_merged, openr1_unique], ignore_index=True)
        print("  Strategy: keep guru first, then append unique openr1 samples")
    elif keep == "openr1":
        df_merged = df_openr1.copy()
        guru_unique = df_guru[~df_guru["user_content"].isin(df_openr1["user_content"])]
        df_merged = pd.concat([df_merged, guru_unique], ignore_index=True)
        print("  Strategy: keep openr1 first, then append unique guru samples")
    else:
        df_merged = pd.concat([df_openr1, df_guru], ignore_index=True)
        df_merged = df_merged.drop_duplicates(subset=["user_content"], keep=keep)
        print(f"  Strategy: pandas drop_duplicates(keep='{keep}')")

    print(f"  - merged: {df_merged.shape[0]} rows, {df_merged.shape[1]} cols")

    print("\n[5/5] Summary")
    print(f"  - raw total: {df_openr1.shape[0] + df_guru.shape[0]}")
    print(f"  - deduplicated: {df_merged.shape[0]}")
    print(f"  - removed duplicates: {df_openr1.shape[0] + df_guru.shape[0] - df_merged.shape[0]}")

    source_counts = df_merged["_source_dataset"].value_counts()
    print("  - source distribution:")
    for source, count in source_counts.items():
        print(f"    {source}: {count} ({count / df_merged.shape[0] * 100:.2f}%)")

    print(f"\nSaving to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Drop the helper dedup field while keeping source tracking metadata.
    df_merged_final = df_merged.drop(columns=["user_content"])
    df_merged_final.to_parquet(output_path, index=False)
    print("Saved.")

    return df_merged_final


if __name__ == "__main__":
    data_root = Path(os.path.expanduser(os.getenv("DATA_ROOT", "~/data")))

    parser = argparse.ArgumentParser(description="Merge openr1 and guru_rl92k datasets with deduplication")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "both"],
        default="both",
        help="Merge mode: train (train only), test (test only), both (train + test)",
    )
    parser.add_argument(
        "--openr1-train",
        type=str,
        default=str(data_root / "openr1" / "train.parquet"),
        help="Path to OpenR1 training parquet",
    )
    parser.add_argument(
        "--guru-train",
        type=str,
        default=str(data_root / "guru_rl92k" / "train" / "math__combined_54.4k.parquet"),
        help="Path to Guru-RL92k training parquet",
    )
    parser.add_argument(
        "--openr1-test",
        type=str,
        default=str(data_root / "openr1" / "test.parquet"),
        help="Path to OpenR1 test parquet",
    )
    parser.add_argument(
        "--guru-test",
        type=str,
        nargs="+",
        default=[
            str(data_root / "guru_rl92k" / "online_eval" / "math__math_500.parquet"),
            str(data_root / "guru_rl92k" / "online_eval" / "math__aime_repeated_8x_240.parquet"),
        ],
        help="Path(s) to Guru-RL92k test parquet files",
    )
    parser.add_argument(
        "--output-train",
        type=str,
        default=str(data_root / "merged_openr1_guru" / "train.parquet"),
        help="Output path for merged training parquet",
    )
    parser.add_argument(
        "--output-test",
        type=str,
        default=str(data_root / "merged_openr1_guru" / "test.parquet"),
        help="Output path for merged test parquet",
    )
    parser.add_argument(
        "--keep",
        type=str,
        choices=["first", "last", "guru", "openr1"],
        default="guru",
        help="Dedup strategy: first/last (pandas), or prioritize guru/openr1",
    )

    args = parser.parse_args()

    if args.mode in ["train", "both"]:
        print("\n" + "=" * 60)
        print("Merging train set")
        print("=" * 60)
        merge_and_deduplicate(args.openr1_train, args.guru_train, args.output_train, args.keep)

    if args.mode in ["test", "both"]:
        print("\n" + "=" * 60)
        print("Merging test set")
        print("=" * 60)
        merge_test_datasets(args.openr1_test, args.guru_test, args.output_test, args.keep)
