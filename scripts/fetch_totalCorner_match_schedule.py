import os
import pandas as pd
import time
import json
from dotenv import load_dotenv
from api_handler import handle_api_request
from difflib import SequenceMatcher

#Load environment variables
load_dotenv()

TOTAL_CORNER_API_TOKEN = os.getenv("TOTAL_CORNER_API_TOKEN")
BASE_URL = "https://api.totalcorner.com/v1"

TEAM_MAPPING_FILE = "/data/totalCorner/team_name_mapping.json"

#Use team name mapping
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'totalCorner', 'team_name_mapping.json'), "r") as f:
    TEAM_NAME_MAPPING = json.load(f)

#League mapping: Kaggle dataset -> TotalCorner format
LEAGUE_MAPPING = {
    "E0": "England Premier League",
    "D1": "Germany Bundesliga I",
    "SP1": "Spain La Liga",
    "F1": "France Ligue 1",
    "I1": "Italy Serie A"
}

#Load existing processed game data
processed_game_data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'aggregated_data.csv'))

#Ensure date column is in correct format...
processed_game_data["date"] = pd.to_datetime(processed_game_data["date"], errors="coerce")
processed_game_data =processed_game_data.dropna(subset=["date"]) 

# Extract unique matches per date, sorted by league
'''
Format:
date {
    league: {{kaggle_id: (home_team, away_team)}, {kaggle_id_2: (home_team2, away_team2)}}
    league2: {{kaggle_id: (home_team, away_team)}, {kaggle_id_2: (home_team2, away_team2)}}
    ...
}
date2 {
    ...
}
'''
match_dict = {}
for _,row in processed_game_data.iterrows():
    date = row["date"].strftime("%Y-%m-%d")
    kaggle_id = row["id_odsp"]
    home_team = row["home_team"]
    away_team =row["away_team"]
    league_code= row["league"]
    
    league_name = LEAGUE_MAPPING.get(league_code)

    #create date if not present
    if date not in match_dict:
        match_dict[date] = {}

    #create league if not present for that date.
    if league_name not in match_dict[date]:
        match_dict[date][league_name] = {}
    
    #Store Kaggle_id as key, with (home_team, away_team) as value
    match_dict[date][league_name][kaggle_id] =(home_team, away_team)

#Print structured match_dict (for debug only)
# for date, leagues in match_dict.items():
#     print(f"📅 Date: {date}")
#     for league, matches in leagues.items():
#         print(f"  🏆 League: {league}")
#         for kaggle_id, teams in matches.items():
#             print(f"    - {kaggle_id}: {teams}")

#--------

def standardize_team_name(league, team_name):
    """Returns the standardised team name if mapping exists... Otherwise, returns the orignal."""
    return TEAM_NAME_MAPPING.get(league, {}).get(team_name, team_name)

# Function to fetch match schedule with pagination and match filtering
def fetch_match_schedule(date):
    """Fetch all matches for a given date."""
    match_ids =[]
    unmatched_ids=[]
    page=1

    #Deep copy remaining matches:
    remaining_matches = {league: matches.copy() for league, matches in match_dict.get(date, {}).items()}
    print(f"remaining_matches:\n{remaining_matches}")
    date_without_dashes=date.replace("-", "")
 
    while True: 
        print(f"Page={page}")
        url = f"{BASE_URL}/match/schedule?token={TOTAL_CORNER_API_TOKEN}&date={date_without_dashes}&page={page}"
        data = handle_api_request(url)

        if data is None:
            #stop on failure
            print(f"Skipping {date} due to API failure.")
            break 
        
        #get 'data' fielkd from API
        match_list = data.get("data", [])
        if not match_list:
            break

        for match in match_list: 
            league_api =match.get("l")

            # Onlly focus on matches wehere league is one of leagues we are looking for. This way it is efficient
            if league_api in remaining_matches:
                #If so, get the id, home team and away team
                totalCorner_id = match.get("id")
                home_team_api=match.get("h") 
                away_team_api= match.get("a")
                print(f"🔍 Checking {league_api}: {home_team_api} vs {away_team_api}")

                match_found = False #flag for if any match was found

                for kaggle_id,(home_team_kaggle, away_team_kaggle) in list(remaining_matches[league_api].items()):
                    home_team_kaggle=standardize_team_name(league_api, home_team_kaggle)
                    away_team_kaggle=standardize_team_name(league_api, away_team_kaggle)
                    
                    #compare if home and away team are equivelent
                    if (home_team_kaggle == home_team_api) and (away_team_kaggle == away_team_api):
                        print(f"✅ Match Found: {home_team_kaggle} vs {away_team_kaggle} ({league_api}) on {date}, ID: {totalCorner_id}")

                        #Add to match_ids list
                        match_ids.append({ 
                            "kaggle_id": kaggle_id,
                            "totalCorner_id": totalCorner_id,
                        }) 

                        # Remove found match
                        del remaining_matches[league_api][kaggle_id]
                        print(f"✅ Remaining matches after removal:\n{remaining_matches}")

                        #If the league has no remaining matches, remove it from remaining_matches...
                        if not remaining_matches[league_api]: 
                            del remaining_matches[league_api] 
                            print(f"🚨 {league_api} removed from remaining_matches as it's now empty.")

                        match_found = True  # A match was found
                        break 
                
                # If no match was found, add to unmatches list
                if not match_found:
                    unmatched_ids.append({
                        "kaggle_id": kaggle_id,
                        "league": league_api,
                        "home_team":home_team_kaggle,
                        "away_team":away_team_kaggle
                    })
                    print(f"❌ No Match Found: {home_team_api} vs {away_team_api} ({league_api}) on {date}")
 
        #stop paginating if there are no remaining matches
        if not remaining_matches:
            break
        pagination = data.get("pagination", {})
        if not pagination.get("next"):
            break
        page += 1
        time.sleep(1)

    return match_ids, unmatched_ids

#--------

all_match_ids = []
all_unmatched_ids = []

for date in match_dict.keys():
    print(f"📅 Fetching match IDs for {date}...")
 
    match_ids, unmatched_ids = fetch_match_schedule(date)
 
    #Save matched results to CSV
    if match_ids: 
        match_id_df = pd.DataFrame(match_ids) 
        file ="data/totalCorner/match_ids.csv" 
        match_id_df.to_csv(file, mode='a',header=not os.path.exists(file),index=False) #use append rather than write, so more can be added in the future
        print(f"✅ Match ID fetching complete for {date}. Appended to match_ids.csv.")

    #Save unmatched results too...
    if unmatched_ids:
        unmatched_df = pd.DataFrame(unmatched_ids) 
        file ="data/totalCorner/unmatched_matches.csv"
        unmatched_df.to_csv(file, mode='a',header=not os.path.exists(file),index=False)
        print(f"⚠ Unmatched matches for {date} appended to unmatched_matches.csv.")
    else: 
        print(f"✅ No unmatched matches found for {date}.") 

 

#---------
'''Script used to extract unique team names in our dataset, which were then used to add name corrections (if applicable) by TotalCorner's API format'''

# # Extract unique team names per league
# team_name_dict = {}
# for _, row in processed_game_data.iterrows():
#     league_code = row["league"]
#     league_name = LEAGUE_MAPPING.get(league_code, league_code)  # Convert league code to full name
#     home_team = row["home_team"]
#     away_team = row["away_team"]

#     # Initialize league entry if not present
#     if league_name not in team_name_dict:
#         team_name_dict[league_name] = set()

#     # Add team names to the league group
#     team_name_dict[league_name].add(home_team)
#     team_name_dict[league_name].add(away_team)

# # Convert sets to sorted lists for easy readability
# sorted_team_names = {league: sorted(list(teams)) for league, teams in team_name_dict.items()}

# # Prepare formatted dictionary for manual editing
# team_mapping_template = {league: {team: "" for team in teams} for league, teams in sorted_team_names.items()}

# # Save to a JSON file
# with open("data/totalCorner/team_name_mapping.json", "w") as f:
#     json.dump(team_mapping_template, f, indent=4)