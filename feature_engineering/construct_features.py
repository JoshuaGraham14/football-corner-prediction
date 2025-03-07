import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from feature_engineering.config_loader import load_config

def calc_home_urgency(df):
    """
    Computes urgency for the home team to attack based on:
    - If they are losing at 80 minutes (and goal diff is not over 2)
    - If the match is drawn-> urgency is based on odds
    """
    urgency = []
    for i in range(len(df)):
        u = 0
        if df.loc[i, "goal_diff_80"]<0 and df.loc[i, "goal_diff_80"]>=-2:  # Home team losing
            # urgency is proportional to inverse of home team's odds .. normalised, by the absolute goal difference
            u = (1 / df.loc[i, "odd_h"]) / abs(df.loc[i,"goal_diff_80"])
        elif df.loc[i, "goal_diff_80"] == 0:  #drawing
            #proportional to inverse of home teams odds... normalised by sum of the inverse of both teams odds
            u = (1 / df.loc[i,"odd_h"]) / ((1 / df.loc[i, "odd_h"]) + (1 / df.loc[i, "odd_a"]))
        urgency.append(round(u, 3)) #else append urgency=0
    return urgency 

def calc_away_urgency(df):
    """
    Computes urgency for the away team to attack based on:
    - If they are losing at 80 minutes (and goal diff is not over 2)
    - If the match is drawn-> urgency is based on odds
    """
    urgency = []
    for i in range(len(df)):
        u = 0
        if df.loc[i, "goal_diff_80"] >0 and df.loc[i, "goal_diff_80"]<=2:
            u = 1 / df.loc[i, "odd_a"] / abs(df.loc[i,"goal_diff_80"])
        elif df.loc[i, "goal_diff_80"] ==0:
            u = (1 / df.loc[i,"odd_a"]) / ((1 / df.loc[i, "odd_h"]) + (1 / df.loc[i, "odd_a"]))
        urgency.append(round(u,3))
    return urgency

def construct_features(df, selected_constructed_features):
    """
    Constructucts features dynamically based on config:
    """
    if not selected_constructed_features:
        return df

    #dictionary mapping to calc each feature
    feature_operations = {
        "total_shots_pre_80": lambda df: df["home_shots_pre80"] +df["away_shots_pre80"],
        "total_fouls_pre_80":lambda df: df["home_fouls_pre80"]+ df["away_fouls_pre80"],
        "total_yellow_cards_pre_80": lambda df: df["home_yellow_cards_pre80"] + df["away_yellow_cards_pre80"],
        "total_sending_off_pre_80": lambda df: df["home_sending_off_pre80"] + df["away_sending_off_pre80"],
        "total_corners_pre_80": lambda df:df["home_corners_pre80"]+ df["away_corners_pre80"],
        "total_corners_70_75": lambda df: df["home_corners_70_75"] + df["away_corners_70_75"],
        "total_shots_70_75":lambda df: df["home_shots_70_75"] +df["away_shots_70_75"],
        "total_fouls_70_75": lambda df: df["home_fouls_70_75"] +df["away_fouls_70_75"],
        "total_corners_75_80": lambda df: df["home_corners_75_80"] + df["away_corners_75_80"],
        "total_shots_75_80": lambda df:df["home_shots_75_80"] +df["away_shots_75_80"],
        "total_fouls_75_80": lambda df: df["home_fouls_75_80"] +df["away_fouls_75_80"],

        "odds_ratio": lambda df: (df["odd_h"] / df["odd_a"]),

        "shot_to_corner_ratio_pre_80": lambda df: (df["total_shots_pre_80"] /df["total_corners_pre_80"]).fillna(0), #..avoid division by zero
        "team_aggression_score_pre_80": lambda df: (df["total_fouls_pre_80"]+df["total_yellow_cards_pre_80"]) /(df["total_shots_pre_80"]).fillna(0),

        "home_urgency_to_attack":lambda df: calc_home_urgency(df),
        "away_urgency_to_attack":lambda df: calc_away_urgency(df),

        "home_momentum_to_attack": lambda df: (
            (df["home_shots_75_80"]-df["home_shots_70_75"]) +
            (df["home_corners_75_80"]-df["home_corners_70_75"])
        ) *df["home_urgency_to_attack"],

        "away_momentum_to_attack": lambda df: (
            (df["away_shots_75_80"]-df["away_shots_70_75"]) +
            (df["away_corners_75_80"]-df["away_corners_70_75"])
        ) *df["away_urgency_to_attack"],

        "attack_intensity": lambda df: (df["goal_diff_80"].abs()==1) * (df["total_shots_75_80"]+df["total_corners_75_80"]),
        "defensive_pressure":lambda df: df["total_fouls_75_80"]-df["total_fouls_70_75"]
    }
    
    #Only aply features selected from config!
    for feature, operation in feature_operations.items():
        if feature in selected_constructed_features:
            df[feature] =operation(df)
    df =df.round(3)
    return df

def prepare_final_dataframe(df):
    """
    Selects relevant columns from the dataset for training dataset
    """
    config = load_config()

    context_features = config.get("features",{}).get("context_features",[])
    selected_features = config.get("features",{}).get("selected_features",[])
    constructed_features = config.get("features",{}).get("constructed_features",[])
    target_variables =config.get("features",{}).get("target_variables",[])

    # Debug:
    # print(f"context_features: {context_features}")
    # print(f"selected_features: {selected_features}")
    # print(f"constructed_features: {constructed_features}")
    # print(f"target_variables: {target_variables}")

    if constructed_features is None:
        constructed_features = []
    df=construct_features(df, constructed_features)
    
    # Select only necessary columns
    selected_columns = context_features + selected_features + constructed_features + target_variables
    df = df[selected_columns]

    #Drop NaN and inf rows:
    df = df.copy()
    df.loc[:, :] = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df, context_features, selected_features, constructed_features, target_variables

def main():
    """
    *Only used for testing*
    - Loads dataset, applies construct_features() func, prepares the final DataFrame, and saves it
    """

    file_path = "data/processed/aggregated_data.csv"
    df = pd.read_csv(file_path)

    # df=construct_features(df)
    df_final,_,_,_,_=prepare_final_dataframe(df)

    # Save it:
    df_final.to_csv("df_engineered.csv", index=False)
    print("df_engineered.csv saved ✅")

if __name__ == "__main__":
    main()
