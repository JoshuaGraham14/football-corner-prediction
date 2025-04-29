""""
Loads the configuration file (config.yaml)
"""

import os
import sys
import yaml
# Set project root
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

def load_config(config_name):
    """
    Loads the passed in config file from the project root folder.
    """
    config_path = os.path.join(project_root, config_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"ERROR: config.yaml not found at {config_path}")
    else:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        print("Config loaded successfully!")
    return config