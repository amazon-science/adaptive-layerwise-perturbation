"""
Custom masking logic for SFT training
Masks:
1. All observation content (Code execution result: ...)
2. All content before and including the last failed turn
"""

import re
from typing import Dict, List
from transformers import PreTrainedTokenizer


def has_error_in_observation(obs: str) -> bool:
    """Check if an observation contains an error"""
    obs_lower = obs.lower().strip()
    error_keywords = ['err', 'error', 'timeout', 'exception', 'traceback']
    
    # Check for error keywords
    if any(keyword in obs_lower for keyword in error_keywords):
        return True
    
    # Check for empty or None observations
    if obs_lower == '' or obs_lower == '[]' or obs_lower == 'none':
        return True
    
    return False


def find_observation_positions(text: str) -> List[tuple]:
    """
    Find all observation positions in the text
    Returns list of (start, end, has_error) tuples
    """
    # Pattern to match "Code execution result: " followed by content until next code block or end
    # More robust pattern that handles different formats
    pattern = r'Code execution result:\s*\n?(.*?)(?=```|$)'
    
    observations = []
    for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
        start = match.start()
        end = match.end()
        obs_content = match.group(1).strip()
        has_error = has_error_in_observation(obs_content)
        observations.append((start, end, has_error))
    
    return observations


def find_last_error_turn(text: str) -> int:
    """
    Find the character position of the end of the last failed turn
    Returns -1 if no errors found
    """
    observations = find_observation_positions(text)
    
    if not observations:
        return -1
    
    # Find the last error observation
    last_error_pos = -1
    for start, end, has_error in observations:
        if has_error:
            last_error_pos = end
    
    return last_error_pos


def mask_labels_custom(
    input_ids: List[int],
    labels: List[int],
    tokenizer: PreTrainedTokenizer,
    assistant_content: str
) -> List[int]:
    """
    Apply custom masking logic to labels
    
    Args:
        input_ids: Token IDs of the input
        labels: Original labels
        tokenizer: Tokenizer used
        assistant_content: The assistant's response content
    
    Returns:
        Modified labels with custom masking applied
    """
    # Decode to get the full text
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    
    # Find where assistant content starts in the full text
    # This is approximate - we'll use a more robust token-based approach
    
    # Strategy: Re-encode the assistant content and find matching positions
    assistant_tokens = tokenizer.encode(assistant_content, add_special_tokens=False)
    
    # Find all observation positions in the assistant content
    observations = find_observation_positions(assistant_content)
    
    # Find last error position
    last_error_pos = find_last_error_turn(assistant_content)
    
    # If there's an error, mask everything up to and including that position
    if last_error_pos != -1:
        # Find the character position in assistant content
        # Then map to token positions
        masked_text = assistant_content[:last_error_pos]
        masked_tokens = tokenizer.encode(masked_text, add_special_tokens=False)
        
        # Find where these tokens appear in input_ids
        # Mask from the start of assistant content
        assistant_start = find_subsequence(input_ids, assistant_tokens[:min(20, len(assistant_tokens))])
        
        if assistant_start != -1:
            # Mask everything from assistant start to the error position
            mask_end = assistant_start + len(masked_tokens)
            for i in range(assistant_start, min(mask_end, len(labels))):
                labels[i] = -100
    
    # Mask all observations regardless of error status
    for obs_start_char, obs_end_char, _ in observations:
        # Get the text of the observation
        obs_text = assistant_content[obs_start_char:obs_end_char]
        obs_tokens = tokenizer.encode(obs_text, add_special_tokens=False)
        
        # Find this sequence in the input_ids
        # Search starting from assistant content
        assistant_start = find_subsequence(input_ids, assistant_tokens[:min(20, len(assistant_tokens))])
        
        if assistant_start != -1:
            # Find observation tokens within assistant section
            obs_token_start = find_subsequence(
                input_ids[assistant_start:], 
                obs_tokens[:min(10, len(obs_tokens))]
            )
            
            if obs_token_start != -1:
                obs_token_start += assistant_start
                obs_token_end = obs_token_start + len(obs_tokens)
                
                # Mask these tokens
                for i in range(obs_token_start, min(obs_token_end, len(labels))):
                    labels[i] = -100
    
    return labels


def find_subsequence(seq: List[int], subseq: List[int]) -> int:
    """
    Find the starting index of a subsequence in a sequence
    Returns -1 if not found
    """
    if not subseq:
        return -1
    
    subseq_len = len(subseq)
    for i in range(len(seq) - subseq_len + 1):
        if seq[i:i+subseq_len] == subseq:
            return i
    
    return -1


def apply_custom_masking(example: Dict, tokenizer: PreTrainedTokenizer) -> Dict:
    """
    Main function to apply custom masking to a training example
    This should be called after tokenization
    """
    # Extract assistant content from messages
    messages = example.get('messages', [])
    assistant_content = None
    
    for msg in messages:
        if msg.get('role') == 'assistant':
            assistant_content = msg.get('content', '')
            break
    
    if assistant_content is None:
        return example
    
    # Apply custom masking if we have tokenized data
    if 'input_ids' in example and 'labels' in example:
        example['labels'] = mask_labels_custom(
            example['input_ids'],
            example['labels'],
            tokenizer,
            assistant_content
        )
    
    return example

