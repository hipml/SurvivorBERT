from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login
from dotenv import load_dotenv

import pandas as pd
import re
import os

def normalize_speaker(speaker):
    if not speaker:
        return None
        
    # Special case for Probst
    if speaker.upper() == 'PROBST':
        return 'Jeff'
    
    # Convert to proper case
    return ' '.join(word.capitalize() for word in speaker.lower().split())

def normalize_text(text):
    # If text is all caps, convert to title case first, then proper case
    if text.isupper():
        text = text.title()
    
    # Handle dialogue with dashes
    if text.startswith('-'):
        text = text.replace('-', '').strip()
    
    # Special handling for show name
    text = text.replace('"SURVIVOR"', 'Survivor')
    text = text.replace('"Survivor"', 'Survivor')
    
    return text.strip()

def extract_speaker(text):
    # Remove ">>" markers
    text = text.replace('>>', '').strip()
    
    # Handle parenthetical notes separately
    parenthetical_match = re.search(r'\((.*?)\)', text)
    parenthetical = parenthetical_match.group(0) if parenthetical_match else ''
    text = text.replace(parenthetical, '').strip()
    
    # Match pattern "Speaker: text" - handles both upper and lower case names
    match = re.match(r'^([A-Za-z][A-Za-z\s&]+?):\s*(.*)', text.strip())
    if match:
        speaker = normalize_speaker(match.group(1))
        content = normalize_text(match.group(2))
        return speaker, content
    return None, normalize_text(text)


def merge_subtitles(df):
    merged_entries = []
    current_sentence = {
        'episode': None,
        'speaker': None,
        'text': [],
        'start_time': None,
        'end_time': None,
        'subtitle_number': None
    }
    
    def is_sentence_end(text):
        return bool(re.search(r'[.!?]$', text.strip()))
    
    def clean_text(text):
        # Remove ">>" markers and clean up spaces
        text = re.sub(r'\s+', ' ', text.replace('>>', '').strip())
        return normalize_text(text)  # Add normalization here
    
    for idx, row in df.iterrows():
        text = clean_text(row['text'])  # Now calls normalize_text
        speaker, content = extract_speaker(text)
        
        if current_sentence['text'] == [] or speaker:
            if current_sentence['text']:
                merged_entries.append({
                    'episode': current_sentence['episode'],
                    'speaker': current_sentence['speaker'],
                    'text': clean_text(' '.join(current_sentence['text'])),  # Add normalization here
                    'start_time': current_sentence['start_time'],
                    'end_time': current_sentence['end_time'],
                    'original_subtitles': current_sentence['subtitle_number']
                })
            
            current_sentence = {
                'episode': row['episode'],
                'speaker': speaker,
                'text': [content],
                'start_time': row['start_time'],
                'end_time': row['end_time'],
                'subtitle_number': [row['subtitle_number']]
            }
        else:
            current_sentence['text'].append(content)
            current_sentence['end_time'] = row['end_time']
            current_sentence['subtitle_number'].append(row['subtitle_number'])
        
        if is_sentence_end(content):
            merged_entries.append({
                'episode': current_sentence['episode'],
                'speaker': current_sentence['speaker'],
                'text': clean_text(' '.join(current_sentence['text'])),  # Add normalization here
                'start_time': current_sentence['start_time'],
                'end_time': current_sentence['end_time'],
                'original_subtitles': current_sentence['subtitle_number']
            })
            current_sentence = {'text': [], 'subtitle_number': [], 'speaker': None}
            
    return pd.DataFrame(merged_entries)

load_dotenv()
token = os.getenv('HF_TOKEN')
login(token)

dataset = load_dataset("hipml/survivor-subtitles")
df = pd.DataFrame(dataset['train'])
merged_df = merge_subtitles(df)

# print("\n=== Sample Processed Entries ===\n")
# samples = merged_df.sample(n=10)
# for _, row in samples.iterrows():
#     print(f"Episode: {row['episode']}")
#     print(f"Speaker: {row['speaker'] or 'None'}")
#     print(f"Text: {row['text']}")
#     print("---")

processed_dataset = Dataset.from_pandas(merged_df)

dataset_dict = DatasetDict({
    "train": processed_dataset
})

dataset_dict.push_to_hub(
    "hipml/survivor-subtitles-processed",
    private=False
)
