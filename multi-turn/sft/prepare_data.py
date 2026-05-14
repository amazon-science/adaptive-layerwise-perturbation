from datasets import load_dataset
import re

# Load dataset
ds = load_dataset('ElonTusk2001/rstar_sft', split='train')

def change_format(example):
    """
    Process response format conversion:
    1. Remove <end_of_step> inside <code>...<end_of_code> and replace with ```python ... ```
    2. Replace <output>...<end_of_output> with Code execution result: ...
    3. Remove outer tags of <answer>...<end_of_answer>, keep only content
    """
    if 'response' not in example or not example['response']:
        return example
    
    response = example['response']
    
    # 1. Process <code>...</code>: remove <end_of_step>, replace with ```python ... ```
    def process_code(match):
        code_content = match.group(1)
        # Remove all <end_of_step>
        code_content = code_content.replace('<end_of_step>\n', '').replace('<end_of_step>\n\n', '\n')
        # Replace with markdown code block format
        return f'```python\n{code_content}\n```'
    
    response = re.sub(r'<code>\s*(.*?)\s*<end_of_code>', process_code, response, flags=re.DOTALL)
    
    # 2. Process <output>...</output>: replace with Code execution result: ...
    def process_output(match):
        output_content = match.group(1)
        return f'Code execution result: {output_content}'
    
    response = re.sub(r'<output>\s*(.*?)\s*<end_of_output>', process_output, response, flags=re.DOTALL)
    
    # 3. Process <answer>...</answer>: remove tags, keep only content
    def process_answer(match):
        answer_content = match.group(1)
        return answer_content.strip()
    
    response = re.sub(r'<answer>\s*(.*?)\s*<end_of_answer>', process_answer, response, flags=re.DOTALL)
    
    example['response'] = response
    return example


# Convert data format: from query/response to messages format
def convert_to_messages(example):
    """
    Convert query/response format to messages format required by axolotl
    Format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    messages = []
    
    # Add user message (from query field)
    if 'query' in example and example['query']:
        messages.append({
            "role": "user",
            "content": example['query']
        })
    
    # Add assistant message (from response field)
    if 'response' in example and example['response']:
        messages.append({
            "role": "assistant",
            "content": example['response']
        })
    
    return {"messages": messages}

# First process response format conversion
ds = ds.map(change_format)

# Then convert to messages format
ds = ds.map(convert_to_messages, remove_columns=ds.column_names)

# Save as JSON Lines format
ds.to_json('sft_data.jsonl')

print(f"Dataset converted and saved to: sft_data.jsonl")
print(f"Dataset size: {len(ds)}  records")
print(f"Example data format:")
print(ds[0])