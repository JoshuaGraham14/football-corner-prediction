import os
import pandas as pd
import time
from dotenv import load_dotenv

#Load environment variables
load_dotenv()

READ_FOLDER_PATH = '/data/processed/snippets/'
API_TOKEN = os.getenv("API_TOKEN")
BASE_URL = "https://api.totalcorner.com/v1"

#Load existing processed game data
processed_game_data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'snippets', 'aggregated_data_snippet.csv'))

#Ensure date column is in correct format...
processed_game_data["date"] = pd.to_datetime(processed_game_data["date"], errors="coerce")
processed_game_data =processed_game_data.dropna(subset=["date"]) 

#Extract unique matches per date
match_dict = {}
for _, row in processed_game_data.iterrows():
    date = row["date"].strftime("%Y-%m-%d")
    home_team = row["home_team"]
    away_team = row["away_team"]
    league =row["league"] 
    
    if date not in match_dict:
        match_dict[date] = set()
    match_dict[date].add((home_team,away_team,league))

print(match_dict)

