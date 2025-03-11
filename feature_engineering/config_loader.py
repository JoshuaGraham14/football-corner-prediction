import os
import yaml
import pprint

# Default hyperparameters used if not specified...
DEFAULT_HYPERPARAMETERS = {
    "random_forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42}
}

def load_config(config_file="config.yaml"):
    """
    Loads configuration from .yaml file:
    """

    # use absolute path to project root...
    project_root =os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path =os.path.join(project_root, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ ERROR: `config.yaml` not found at {config_path}")

    # Load YAML file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    #Ensure optimise_hyperparameters is always set :
    config["model"]["binary"].setdefault("grid_search", False)
    config["model"]["continuous"].setdefault("grid_search", False)

    #fill missing hyperparameters with defaults...
    for model_type in ["binary", "continuous"]:
        for model in config["model"][model_type]["models"]:
            config["model"][model_type]["hyperparameters"].setdefault(model, DEFAULT_HYPERPARAMETERS.get(model,{}))
    return config

#(for testing)...
if __name__ == "__main__":
    config = load_config()
    pprint.pprint(config, sort_dicts=False)
