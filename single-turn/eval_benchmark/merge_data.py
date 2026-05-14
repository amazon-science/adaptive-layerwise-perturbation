import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

"""
If we use multiple VLLM processes to accelerate the generation, we need to use this script to merge them.
"""


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    base_path: Optional[str] = field(
        default="/home/xiongwei/gshf/data/gen_data",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    num_datasets: Optional[int] = field(
        default=8,
        metadata={"help": "the location of the output file"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


all_dirs = [script_args.base_path + str(i) + ".json" for i in range(script_args.num_datasets)]

gathered_data = []
for my_dir in all_dirs:
    if not os.path.isfile(my_dir):
        raise FileNotFoundError(
            f"Expected data file not found: {my_dir}. "
            "gen_data.py may have failed (e.g. vLLM model load error). Check logs for the failing step."
        )

    local_count = 0
    with open(my_dir, "r", encoding="utf8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {my_dir} at line {line_idx}: {e}") from e
            gathered_data.append(sample)
            local_count += 1

    print(local_count)

random.shuffle(gathered_data)

print("I collect ", len(gathered_data), "samples")

with open(script_args.output_dir, "w", encoding="utf8") as f:
    for i in range(len(gathered_data)):
        json.dump(gathered_data[i], f, ensure_ascii=False)
        f.write('\n')
