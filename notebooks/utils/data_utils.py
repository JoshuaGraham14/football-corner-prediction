import os
import pandas as pd
import numpy as np

def load_dataset(config):
    """
    Loads the dataset from the path specified in the config file.
    """
    
    READ_FOLDER_PATH = config["paths"]["dataset"]
    if not os.path.exists(READ_FOLDER_PATH):
        raise FileNotFoundError(f"ERROR: Dataset not found at {READ_FOLDER_PATH}")
    
    df = pd.read_csv(READ_FOLDER_PATH)
    print(f"📁 Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    return df

def preprocess_data(df, config):
    """
    Selects specified features from the dataset, removes NaN and infinite values.
    """
    pd.set_option("display.max_colwidth", None) 

    context_features = config.get("features",{}).get("context_features",[])
    selected_features = config.get("features",{}).get("selected_features",[])
    constructed_features = config.get("features",{}).get("constructed_features",[])
    target_variable =config.get("features",{}).get("target_variable",[])
    # possible_target_variables =config.get("features",{}).get("possible_target_variables",[])

    if constructed_features is None:
        constructed_features = []

    selected_columns = context_features + selected_features + constructed_features + target_variable
    df = df[selected_columns]

    #Drop NaN and inf rows:
    df_selected = df.copy()
    df_selected.loc[:, :] = df.replace([np.inf, -np.inf], np.nan)
    df_selected = df_selected.dropna()

    print(f"✅ Dataset Preprocessed: {df_selected.shape[0]} rows, {df_selected.shape[1]} columns")

    return df_selected, selected_features, constructed_features, target_variable[0]
