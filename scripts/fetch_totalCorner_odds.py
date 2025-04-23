import os
import pandas as pd
import scipy.stats as stats
import time

from api_handler import handle_api_request

TOTAL_CORNER_API_TOKEN = os.getenv("TOTAL_CORNER_API_TOKEN")
BASE_URL = "https://api.totalcorner.com/v1"

def fetch_total_corner_data(match_id):
    url = f"{BASE_URL}/match/odds/{match_id}?token={TOTAL_CORNER_API_TOKEN}&columns=cornerList"
    data =handle_api_request(url) #use my handle_api_request func

    #check for api errors
    if data is None:
        print(f"❌ API failure for match_id: {match_id}")
        return None
    match_odds = data.get("data", [])
    if not match_odds:
        print(f"*EMPTY*: No data found for match_id: {match_id}")

    return match_odds

#Func: extract the earliest available odds for 80min
def extract_earliest_80th_minute_data(match_data):
    if not match_data or not isinstance(match_data, list):
        print(f"Invalid match_data format: {match_data}")
        return None

    #get corner list data
    match = match_data[0]
    corner_list = match.get("corner_list", [])
    if not corner_list:
        print(f"No 'corner_list' found in match data")
        return None

    #Filter entries for only 80th minute
    min_80_entries = [entry for entry in corner_list if entry[0]=="80"]
    if not min_80_entries:
        print(f"No 80th-minute data found")
        return None

    #sort by timestamp (field #4)...
    min_80_entries.sort(key=lambda x: x[4])
    print(f"-> Earliest 80th-minute odds found: {min_80_entries[0]}")
    return min_80_entries[0] #return first one (i.e. earlist)

#Converts decimal odds to implied probability
def implied_probability(decimal_odds):
    return 1/decimal_odds

#Estimate expected total corners
def expected_corners(market_line, over_odds, under_odds):
    over_prob=implied_probability(over_odds)
    under_prob=implied_probability(under_odds)
    return market_line - (over_prob-under_prob)

#Func: Calculates probability of 1+ corner using poisson distr
def odds_1_plus_corner(expected_additional):
    #Poission distr wiht: P(0, λ)
    prob_0_corners = stats.poisson.pmf(0, expected_additional) 
    prob_1_plus = 1 - prob_0_corners #P(1+ corners) = 1- P(0 corners)
    
    return round(1/prob_1_plus if prob_1_plus>0 else 100, 2)


#----- MAIN -----
OUTPUT_FILE = "data/totalCorner/totalCorner_odds.csv"

#Read match_ids.csv
print("Loading match_ids.csv...")
match_ids_df = pd.read_csv("data/totalCorner/match_ids.csv")
print(f"✅ Loaded {len(match_ids_df)} match IDs")

match_count=len(match_ids_df)
processed_count=0
file_exists=os.path.exists(OUTPUT_FILE) #check if file exists

for index,row in match_ids_df.iterrows():
    totalCorner_id=row["totalCorner_id"]
    kaggle_id=row["kaggle_id"]

    print(f"\n🔍 Processing match {processed_count+1}/{match_count} (TotalCorner ID: {totalCorner_id})...")

    #skip any bad data
    match_data = fetch_total_corner_data(totalCorner_id)
    if not match_data:
        print(f"~ Skipping match {totalCorner_id} due to missing data")
        continue
    earliest_80th_min_data = extract_earliest_80th_minute_data(match_data)
    if not earliest_80th_min_data:
        print(f"~ No valid 80th-minute data for match {totalCorner_id} - Skipping...")
        continue

    #Get required data from earliest 80th min data 
    _, market_line, over_odds, under_odds, _, corners_home,corners_away = earliest_80th_min_data
    market_line, over_odds, under_odds = map(float,[market_line, over_odds, under_odds])
    corners_home, corners_away = int(corners_home), int(corners_away)

    #compute expected_corners using over and under odds
    total_corners=corners_home +corners_away
    expected_total=expected_corners(market_line, over_odds, under_odds)
    expected_additional = max(0,round(expected_total-total_corners, 2))
    odds_1_plus= odds_1_plus_corner(expected_additional)

    #store in PD df...
    result=pd.DataFrame([{
        "kaggle_id":kaggle_id,
        "totalCorner_id":totalCorner_id,
        "market_line": market_line,
        "over_odds": over_odds,
        "under_odds": under_odds,
        "corners_home":corners_home,
        "corners_away": corners_away,
        "total_corners": total_corners,
        "expected_total_corners": round(expected_total, 2),
        "expected_additional_corners":expected_additional,
        "odds_1_plus_corner": odds_1_plus
    }])

    #append to csv after each match...
    result.to_csv(OUTPUT_FILE, mode='a', header=not file_exists, index=False)
    file_exists = True  #ensure header isnt written again

    processed_count += 1
    print(f"✅ Processed match {totalCorner_id} ({processed_count}/{match_count}) and appended to CSV.")

    #small delay to avoid limits
    time.sleep(1)

print("\n✅ Finishd processing all matches and data saved in processed_totalcorner_data.csv.")
