from datasets import load_dataset
import re
from collections import Counter
from typing import Dict, List, Tuple

def create_label_patterns() -> Dict[str, List[str]]:
    return {
        "tribal_council": {
            "voting": [
                "it is time to vote",
                "i'll go tally the votes",
                "you may now vote",
                "one by one"
            ],
            "vote_reading": [
                "the person voted out",
                "bring me your torch",
                "the tribe has spoken",
                "time for you to go",
                "once the votes are read",
                "i'll read the votes"
            ],
            "discussion": [
                "why do you deserve",
                "why should you stay",
                "final tribal council",
                "jury",
                "make your case"
            ]
        },
        "challenge": {
            "setup": [
                "survivors ready",
                "immunity is up for grabs",
                "come on in guys",
                "come on in",
                "wanna know what you're playing for",
                "worth playing for",
                "for today's challenge",
                "first tribe to"
            ],
            "action": [
                "go!",
                "for immunity",
                "immunity is back up for grabs",
                "taking the lead",
                "falling behind",
                "neck and neck",
                "working on",
                "making quick work"
            ],
            "reward": [
                "reward challenge",
                "drop your buffs",
                "worth playing for",
                "guaranteed"
            ]
        },
        "camp_life": {
            "strategy": [
                "alliance",
                "voting bloc",
                "target",
                "vote out",
                "numbers",
                "trust",
                "final three",
                "final two"
            ],
            "personal": [
                "my story",
                "back home",
                "my family",
                "day 1"
            ]
        }
    }

def label_line(text: str, patterns: Dict) -> Tuple[str, str, str, float]:
    """
    Label a single line of dialogue using our heuristics.
    Returns (coarse_category, main_category, sub_category, confidence)
    """
    text = text.lower().strip()
    
    for main_cat, subcats in patterns.items():
        for subcat, phrases in subcats.items():
            for phrase in phrases:
                if phrase in text:
                    # Map to coarse category
                    coarse_cat = {
                        "tribal_council": "tribal",
                        "challenge": "challenge",
                        "camp_life": "camp"
                    }.get(main_cat, "unknown")
                    
                    confidence = 1.0 if phrase == text else 0.8
                    return coarse_cat, main_cat, subcat, confidence
    
    return "unknown", "unknown", "unknown", 0.0

def process_dataset():
    dataset = load_dataset("hipml/survivor-subtitles-cleaned")
    patterns = create_label_patterns()
    
    # First, let's collect all lines in each episode with their timestamps
    episodes = {}
    for item in dataset['train']:
        ep = item['episode']
        if ep not in episodes:
            episodes[ep] = []
        episodes[ep].append({
            'text': item['text'],
            'start_time': item['start_time'],
            'end_time': item['end_time'],
            'index': len(episodes[ep])
        })
    
    def process_episode(episode_lines):
        labels = []
        confidences = []
        
        # First pass: Get high-confidence labels
        for i, line in enumerate(episode_lines):
            coarse, _, _, conf = label_line(line['text'], patterns)
            labels.append(coarse)
            confidences.append(conf)
        
        # Second pass: Use context windows
        WINDOW_SIZE = 30  # number of lines to look before/after
        TIME_WINDOW = 120  # seconds
        
        for i in range(len(episode_lines)):
            if confidences[i] < 0.8:  # Only modify low-confidence labels
                # Look at nearby high-confidence labels
                start_idx = max(0, i - WINDOW_SIZE)
                end_idx = min(len(episode_lines), i + WINDOW_SIZE)
                
                current_time = episode_lines[i]['start_time']
                
                # Count nearby labels within time window
                nearby_labels = []
                for j in range(start_idx, end_idx):
                    time_diff = abs(episode_lines[j]['start_time'] - current_time)
                    if time_diff <= TIME_WINDOW and confidences[j] > 0.8:
                        nearby_labels.append(labels[j])
                
                # If we have nearby high-confidence labels, use the most common one
                if nearby_labels:
                    from collections import Counter
                    most_common = Counter(
                        [l for l in nearby_labels if l != "unknown"]
                    ).most_common(1)
                    if most_common:
                        labels[i] = most_common[0][0]
                        confidences[i] = 0.6  # Lower confidence for context-based labels
        
        return labels, confidences
    
    all_labels = []
    all_confidences = []
    
    # Process each episode
    for ep_lines in episodes.values():
        ep_labels, ep_confidences = process_episode(ep_lines)
        all_labels.extend(ep_labels)
        all_confidences.extend(ep_confidences)
    
    # Create the final dataset
    return dataset['train'].add_column('coarse_label', all_labels).add_column('label_confidence', all_confidences)

if __name__ == "__main__":
    labeled_data = process_dataset()
    
    # Print statistics
    total_lines = len(labeled_data)
    label_counts = Counter(labeled_data['coarse_label'])
    
    print("\nLabel Distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_lines) * 100
        print(f"{label}: {count} ({percentage:.2f}%)")
    
    # Print some examples from each category
    print("\nExample Lines by Category:")
    categories = ['challenge', 'tribal', 'camp']
    for category in categories:
        print(f"\n{category.upper()} Examples (high confidence):")
        shown = 0
        for item in labeled_data:
            if item['coarse_label'] == category and item['label_confidence'] > 0.8 and shown < 3:
                print(f"Episode: {item['episode']}")
                print(f"Text: {item['text']}")
                print(f"Confidence: {item['label_confidence']}")
                shown += 1
