import re
from bs4 import BeautifulSoup

with open("data/subs.html", "r", encoding="utf-8") as file:
    html_content = file.read()

soup = BeautifulSoup(html_content, "html.parser")
pattern = r"(?:season-(\d+)|imdbid-(\d+))$"

season_episode_map = []
current_season = None
episode_number = 1

for a_tag in soup.find_all("a", href=True):
    href = a_tag['href']
    if '/download/' in href:
        href = href.split("view-source:")[1]  # clean up the link

        match = re.search(pattern, href)
        if match:
            season_number = match.group(1)
            imdb_id = match.group(2)

            if season_number:  
                current_season = int(season_number)  
                episode_number = 1  
            elif imdb_id and current_season is not None:
                # if we have an IMDB ID and a current season
                # Format as S01E01 => imdbid or season-based number
                season_episode_map.append(f"S{current_season:02}E{episode_number:02} => {imdb_id}")
                episode_number += 1  

for entry in season_episode_map:
    print(entry)
