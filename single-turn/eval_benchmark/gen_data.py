#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams
import json


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="RLHFlow/test_generation_2k",
        metadata={"help": "single dataset or comma-separated dataset list"},
    )
    local_index: Optional[int] = field(
        default=999,
        metadata={"help": "the local index of the agent"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=4,
        metadata={"help": "the total number of the agents"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=4096,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=4096,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="context_messages",
        metadata={"help": "the key of the dataset"},
    )
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})


def main() -> None:
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    model_path = script_args.model_name_or_path
    print("model_path", model_path)
    seed = script_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        max_model_len=script_args.max_input_length,
        load_format="auto",
        seed=42,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        top_p=1.0,
        max_tokens=script_args.max_new_tokens,
        n=script_args.K,
        stop_token_ids=[tokenizer.eos_token_id] + script_args.eos_ids,
    )

    dataset_names = [name.strip() for name in script_args.dataset_name_or_path.split(",") if name.strip()]
    if len(dataset_names) == 0:
        raise ValueError("dataset_name_or_path is empty after parsing")

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    def make_prompt(example):
        question = example["problem"]
        question = question + " " + instruction_following
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
        }

    gathered_data = []
    for dataset_name in dataset_names:
        ds = load_dataset(dataset_name, split="train")
        ds = ds.map(make_prompt)

        data_size = len(ds["prompt"])
        one_num_share = int(data_size / script_args.my_world_size)
        start = script_args.local_index * one_num_share
        end = (script_args.local_index + 1) * one_num_share
        ds = ds.select(np.arange(start, end))

        print([start, end], dataset_name)
        print(ds, dataset_name)
        if len(ds) == 0:
            continue
        print(ds[0])

        prompts = ds["prompt"]
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

        for i, output in enumerate(outputs):
            tmp_data = {
                "dataset_name": dataset_name,
                "prompt": ds[i]["prompt"],
                "gt": ds[i]["gt"],
                "responses": [out.text for out in output.outputs],
            }
            gathered_data.append(tmp_data)

    print("I collect ", len(gathered_data), "samples")

    with open(script_args.output_dir + str(script_args.local_index) + ".json", "w", encoding="utf8") as f:
        for i in range(len(gathered_data)):
            json.dump(gathered_data[i], f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
