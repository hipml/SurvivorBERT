import configparser
import os
import re
import time
from opensubtitlescom import OpenSubtitles
from typing import Dict, List, Optional


def load_credentials():
    """
    Load OpenSubtitles credentials from config file,
    .opensubtitles_login 
    """

    config = configparser.ConfigParser()
    config_path = '.opensubtitles_login'
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Credentials file not found: {config_path}")
        
    config.read(config_path)
    
    try:
        return {
            'api_key': config['DEFAULT']['API_KEY'],
            'app_name': config['DEFAULT']['APP_NAME'],
            'username': config['DEFAULT']['USERNAME'],
            'password': config['DEFAULT']['PASSWORD']
        }
    except KeyError as e:
        raise KeyError(f"Missing required credential in config file: {e}")


def load_imdb_mappings(filepath):
    """
    Load the episode to IMDB ID mappings from file
    """

    mappings = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if '=>' in line:
                    episode, imdb_id = line.strip().split('=>')
                    mappings[episode.strip()] = imdb_id.strip()
    except FileNotFoundError:
        print(f"Error: Mapping file not found at {filepath}")
        exit(1)
    return mappings

def generate_episode_list():
    """
    Generate list of episodes from S01E01 to S47E14
    """

    episodes = []
    for season in range(1, 48):  # Seasons 1-47
        max_episode = 14 if season == 47 else 13  # Special case for season 47
        for episode in range(1, max_episode + 1):
            episodes.append(f"S{season:02d}E{episode:02d}")
    return episodes

def parse_episode_number(episode_code):
    """
    Parse season and episode numbers from episode code ('S01E01')
    """

    match = re.match(r'S(\d+)E(\d+)', episode_code, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid episode code format: {episode_code}")
    return int(match.group(1)), int(match.group(2))

def download_subtitle(subtitles, imdb_id, season, episode, output_path):
    """
    Download subtitle for a given episode using the OpenSubtitles API
    """

    try:
        response = subtitles.search(
            query="Survivor",
            imdb_id=imdb_id,
            # season_number=season,
            # episode_number=episode,
            languages="en"
        )
        
        if not response.data:
            print(f"No subtitles found for IMDB ID: {imdb_id}")
            return False
            
        srt_content = subtitles.download_and_parse(response.data[0])
        
        formatted_srt = ""
        for subtitle in srt_content:
            def format_timestamp(td):
                total_seconds = int(td.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                milliseconds = td.microseconds // 1000
                return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
            
            formatted_srt += f"{subtitle.index}\n"
            formatted_srt += f"{format_timestamp(subtitle.start)} --> {format_timestamp(subtitle.end)}\n"
            formatted_srt += f"{subtitle.content}\n\n"
            
        # Save the formatted subtitle content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_srt)
            
        return True
        
    except Exception as e:
        print(f"Error downloading subtitle: {e}")
        return False

def generate_failed_episodes():
    """
    Generate list of specific failed episodes
    """
    failed_episodes = []
    
    # Season 17
    failed_episodes.extend([f"S17E{ep:02d}" for ep in [1, 5, 6]])
    
    # Season 18
    failed_episodes.append("S18E04")
    
    # Season 19
    failed_episodes.append("S19E06")
    
    # Season 20
    failed_episodes.append("S20E11")
    
    # Season 21
    failed_episodes.extend([f"S21E{ep:02d}" for ep in [3, 4, 5, 10, 11, 12]])
    
    # Season 22
    failed_episodes.extend([f"S22E{ep:02d}" for ep in [5, 6]])
    
    # Season 24
    failed_episodes.extend([f"S24E{ep:02d}" for ep in [4, 5]])
    
    # Seasons with missing episodes from X onwards
    season_ranges = {
        25: (10, 14),  # Episodes 10+
        27: (8, 14),   # Episodes 8+
        35: (10, 14),  # Episodes 10+
        37: (6, 14),   # Episodes 6+
        44: (11, 14)   # Episodes 11+
    }
    
    for season, (start, end) in season_ranges.items():
        failed_episodes.extend([f"S{season:02d}E{ep:02d}" for ep in range(start, end + 1)])
    
    return failed_episodes

def main():
    credentials = load_credentials()
    subtitles = OpenSubtitles(credentials['app_name'], credentials['api_key'])

    try:
        subtitles.login(credentials['username'], credentials['password'])
    except Exception as e:
        print(f"Failed to login to OpenSubtitles: {e}")
        return
    
    os.makedirs('subtitles', exist_ok=True)
    mappings = load_imdb_mappings('data/imdbid_map.txt')
    episodes = generate_failed_episodes()
    # episodes = generate_episode_list()
    
    total = len(episodes)
    successful = 0
    failed = []
    
    print(f"Starting download of {total} episodes...")
    
    try:
        for episode in episodes:
            if episode not in mappings:
                print(f"Warning: No IMDB ID found for {episode}")
                failed.append(episode)
                continue
                
            imdb_id = mappings[episode]
            output_path = f"subtitles/{episode}.srt"
            
            # parse season and episode numbers
            season, ep_num = parse_episode_number(episode)
            
            print(f"Downloading {episode} (IMDB ID: {imdb_id})...")
            
            if download_subtitle(subtitles, imdb_id, season, ep_num, output_path):
                successful += 1
                print(f"Successfully downloaded {episode}")
            else:
                failed.append(episode)
            
            # add a small delay to avoid hitting API rate limits
            time.sleep(2)
            
    finally:
        try:
            subtitles.logout()
        except:
            pass
    
    print("\nDownload Summary:")
    print(f"Total episodes: {total}")
    print(f"Successfully downloaded: {successful}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed episodes:")
        for episode in failed:
            print(episode)

if __name__ == "__main__":
    main()
