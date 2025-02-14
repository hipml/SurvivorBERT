from datasets import load_dataset, Dataset
import pandas as pd
import re
from tqdm import tqdm
from huggingface_hub import HfApi

def get_continuation_patterns():
    """Return patterns that suggest text continuation."""
    return [
        r'^(and|or|but|nor|for|yet|so|because|while|although|unless|if|than|that|who|which|where)\b',
        r'^[a-z]',
        r'^\s*[,;]\s*',
        r'[,;]\s*$',
        r'\b(to|in|on|at|by|for|with|about|into|through|beyond|over|under)\s*$',
        r'\b(the|a|an|my|your|his|her|their|our|its)\s*$',
    ]

def should_combine_subtitles(current_text, next_text, current_end_time, next_start_time, max_time_gap=1.0):
    """Determine if two subtitle entries should be combined."""
    if not current_text or not next_text:
        return False

    current_text = current_text.strip()
    next_text = next_text.strip()
    
    ends_with_terminal = bool(re.search(r'[.!?][\'")\]]?\s*$', current_text))
    if ends_with_terminal:
        return False
    
    time_gap = next_start_time - current_end_time
    if time_gap > max_time_gap:
        return False
    
    patterns = get_continuation_patterns()
    
    for pattern in patterns:
        if re.search(pattern, next_text, re.IGNORECASE):
            return True
    
    incomplete_patterns = [
        r'\b(and|or|but|if|while|because|that)\s*$',
        r'\b(in|on|at|by|for|with|to|from)\s*$',
        r'[,;]\s*$',
        r'\b(the|a|an)\s*$',
    ]
    
    for pattern in incomplete_patterns:
        if re.search(pattern, current_text, re.IGNORECASE):
            return True
    
    # Check for quotes
    if current_text.count('"') % 2 == 1 or current_text.count('"') % 2 == 1:
        return True
        
    # Check for parentheses/brackets balance
    if (current_text.count('(') > current_text.count(')') or 
        current_text.count('[') > current_text.count(']')):
        return True
    
    return False

def combine_subtitle_fragments(df, max_time_gap=1.0):
    """Combine fragmented subtitle entries into complete sentences."""
    df = df.sort_values(['episode', 'start_time'])
    
    combined_entries = []
    current_entry = None
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Combining subtitles"):
        if current_entry is None:
            current_entry = dict(row)
            continue
            
        if current_entry['episode'] != row['episode']:
            combined_entries.append(current_entry)
            current_entry = dict(row)
            continue
            
        if should_combine_subtitles(current_entry['text'], row['text'], current_entry['end_time'], row['start_time'], max_time_gap):
            current_entry['text'] = f"{current_entry['text'].strip()} {row['text'].strip()}"
            current_entry['end_time'] = row['end_time']
            current_entry['duration'] = current_entry['end_time'] - current_entry['start_time']
        else:
            combined_entries.append(current_entry)
            current_entry = dict(row)
    
    if current_entry is not None:
        combined_entries.append(current_entry)
    
    result_df = pd.DataFrame(combined_entries)
    
    result_df['subtitle_number'] = result_df.groupby('episode').cumcount() + 1
    
    return result_df

def main():
    print("Loading dataset...")
    dataset = load_dataset("hipml/survivor-subtitles", use_auth_token=False)
    print("Dataset loaded!")

    print("Converting to DataFrame...")
    df = pd.DataFrame(dataset['train'])
    print(f"Original number of entries: {len(df)}")
    
    processed_df = combine_subtitle_fragments(df)
    print(f"Number of entries after combining: {len(processed_df)}")
    
    print("Saving local backup...")
    processed_df.to_csv("survivor_subtitles_cleaned.csv", index=False)
    
    print("Converting to Hugging Face dataset format...")
    cleaned_dataset = Dataset.from_pandas(processed_df)
    
    print("Uploading to Hugging Face Hub...")
    cleaned_dataset.push_to_hub(
        "hipml/survivor-subtitles-cleaned",
        private=False,
        commit_message="Add cleaned version of Survivor subtitles with combined sentence fragments"
    )
    
    print("\nProcessing complete!")
    print("Dataset has been uploaded to: https://huggingface.co/datasets/hipml/survivor-subtitles-cleaned")
    
    print("\nStatistics:")
    print(f"Original number of subtitles: {len(df)}")
    print(f"Number of subtitles after combining: {len(processed_df)}")
    print(f"Reduction: {((len(df) - len(processed_df)) / len(df) * 100):.2f}%")
    
if __name__ == "__main__":
    main()
