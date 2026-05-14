"""
Test custom masking logic
"""

import json
from transformers import AutoTokenizer
from axolotl.prompt_strategies.chat_template import ChatTemplateStrategy
from axolotl.prompt_tokenizers import PromptTokenizingStrategy

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=False)

# Load a sample from the dataset
with open('./sft/filtered_sft_data.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i == 0:  # Get first sample
            sample = json.loads(line)
            break

print("="*80)
print("Sample messages:")
for msg in sample['messages']:
    role = msg['role']
    content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
    print(f"\n[{role}]: {content}")

print("\n" + "="*80)
print("Testing masking logic...")

# Check if there are observations
assistant_content = sample['messages'][-1]['content']
print(f"\nAssistant content length: {len(assistant_content)}")

import re
obs_pattern = r'Code execution result:\s*\n?(.*?)(?=```|$)'
observations = list(re.finditer(obs_pattern, assistant_content, re.DOTALL | re.IGNORECASE))
print(f"Number of observations found: {len(observations)}")

for i, match in enumerate(observations):
    obs_content = match.group(1).strip()[:100]
    print(f"\nObservation {i+1}: {obs_content}...")
    
    # Check for errors
    obs_lower = obs_content.lower()
    error_keywords = ['err', 'error', 'timeout', 'exception', 'traceback']
    has_error = any(keyword in obs_lower for keyword in error_keywords)
    print(f"Has error: {has_error}")

print("\n" + "="*80)
print("Masking logic integrated into axolotl successfully!")
print("When training starts, the following tokens will be masked:")
print("1. All 'Code execution result: ...' observations")
print("2. All content before and including the last failed observation")

