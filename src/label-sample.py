from datasets import load_dataset
import random
from collections import Counter
import csv

from label import process_dataset

def sample_lines():
    # Load dataset
    dataset = load_dataset("hipml/survivor-subtitles-cleaned")
    
    # Process using existing labeling function
    labeled_data = process_dataset()
    
    # Calculate proportional samples but with a minimum of 100 per category
    total_samples = 1000
    label_counts = Counter(labeled_data['coarse_label'])
    total_lines = len(labeled_data)
    
    # Sample size per category (minimum 100 for non-unknown)
    samples_per_category = {
        'unknown': 400,  # Fixed sample for unknown
        'challenge': 200,  # Sample more from each known category
        'camp': 200,
        'tribal': 200
    }
    
    # Sample lines for each category
    sampled_lines = []
    for category, sample_size in samples_per_category.items():
        category_lines = [
            {
                'episode': item['episode'],
                'text': item['text'],
                'label': item['coarse_label'],
                'confidence': item['label_confidence'],
                'start_time': item['start_time']
            }
            for item in labeled_data
            if item['coarse_label'] == category
        ]
        
        # Sample with replacement if we need more than available
        if len(category_lines) < sample_size:
            sampled = random.choices(category_lines, k=sample_size)
        else:
            sampled = random.sample(category_lines, k=sample_size)
        
        sampled_lines.extend(sampled)
    
    # Shuffle all sampled lines
    random.shuffle(sampled_lines)
    
    # Write to CSV
    with open('sampled_lines.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'text', 'label', 'confidence', 'start_time'])
        writer.writeheader()
        writer.writerows(sampled_lines)
    
    # Print statistics
    print("\nSampling Summary:")
    sample_counts = Counter(line['label'] for line in sampled_lines)
    for label, count in sample_counts.items():
        original_count = label_counts[label]
        original_percent = (original_count / total_lines) * 100
        sample_percent = (count / len(sampled_lines)) * 100
        print(f"\n{label}:")
        print(f"  Original: {original_count} ({original_percent:.2f}%)")
        print(f"  Sampled:  {count} ({sample_percent:.2f}%)")
        
        # Print first 3 examples from each category
        print(f"\n  Example lines from {label}:")
        examples = [line for line in sampled_lines if line['label'] == label][:3]
        for ex in examples:
            print(f"    - {ex['text']} (conf: {ex['confidence']:.2f})")

if __name__ == "__main__":
    sample_lines()
