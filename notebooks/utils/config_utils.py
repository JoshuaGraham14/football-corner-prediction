""""
Loads the configuration file (config.yaml)
"""

import os
import sys
import yaml
# Set project root
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

#get config file
config_path = os.path.join(project_root, "config.yaml")

def load_config():
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"ERROR: config.yaml not found at {config_path}")
    else:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        print("Config loaded successfully!")
    return config