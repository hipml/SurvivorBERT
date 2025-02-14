import os
import re
from pathlib import Path
import json
from datasets import Dataset
from huggingface_hub import login
from tqdm import tqdm
from dotenv import load_dotenv

def parse_time(time_str):
    """
    Convert SRT timestamp to seconds
    """

    hours, minutes, seconds = time_str.replace(',', '.').split(':')
    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

def parse_srt_file(file_path):
    """
    Parse a single SRT file into a list of dialogue entries
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # split into blocks
    blocks = re.split(r'\n\n+', content.strip())
    dialogues = []
    
    for block in blocks:
        lines = block.split('\n')
        if len(lines) < 3:  # skip invalid blocks
            continue
            
        try:
            # 1 - Parse subtitle number
            subtitle_number = int(lines[0])
            
            # 2 - Parse timestamp
            timestamp_line = lines[1]
            start_time, end_time = timestamp_line.split(' --> ')
            
            # 3- Convert timestamps to seconds
            start_seconds = parse_time(start_time)
            end_seconds = parse_time(end_time)
            
            # 4 - Join remaining lines as text & remove HTML tags
            text = ' '.join(lines[2:])
            text = re.sub(r'<[^>]+>', '', text)  
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 5 - Episode num ("S01E07")
            episode_info = os.path.basename(file_path).split('.')[0]
            
            dialogues.append({
                'episode': episode_info,
                'subtitle_number': subtitle_number,
                'start_time': start_seconds,
                'end_time': end_seconds,
                'duration': end_seconds - start_seconds,
                'text': text
            })
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing block in {file_path}: {block}")
            continue
            
    return dialogues

def process_all_srt_files(directory="./subtitles"):
    """
    Process all SRT files in given directory
    """
    all_dialogues = []
    srt_files = list(Path(directory).glob("S[0-9][0-9]E[0-9][0-9].srt"))
    
    print(f"Starting to process {len(srt_files)} files...")
    
    for srt_file in tqdm(sorted(srt_files), desc="Processing SRT files"):
        dialogues = parse_srt_file(srt_file)
        all_dialogues.extend(dialogues)
    
    print(f"\nTotal dialogues collected: {len(all_dialogues)}")
    print(f"Sample of first dialogue: {all_dialogues[0]}")
    print(f"Sample of last dialogue: {all_dialogues[-1]}")
    
    print("\nCreating HuggingFace Dataset...")
    try:
        dataset = Dataset.from_list(all_dialogues)
        print("Dataset created successfully!")
        print(f"Dataset size: {len(dataset)}")
        
        try:
            dataset.save_to_disk("subtitle_dataset")
        except Exception as e:
            print(f"Error saving dataset: {str(e)}")
            
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        return None
    
    return dataset


def upload_to_hub(dataset, repo_name, private=True):
    load_dotenv()
    token = os.getenv('HF_TOKEN')
    login(token)
    
    dataset.push_to_hub(
        repo_name,
        private=private,
    )

    print(f"\nDataset uploaded to: https://huggingface.co/datasets/{repo_name}")

if __name__ == "__main__":
    dataset = process_all_srt_files()
    print("\nDataset statistics:")
    print(dataset)
    
    # upload to Hub
    repo_name = "hipml/survivor-subtitles" 
    upload_to_hub(dataset, repo_name, private=False)
