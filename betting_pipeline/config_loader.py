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
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    #Ensure optimise_hyperparameters is always set :
    config["model"]["binary"].setdefault("optimise_hyperparameters", False)
    config["model"]["continuous"].setdefault("grid_optimise_hyperparameterssearch", False)

    #fill missing hyperparameters with defaults...
    for model_type in ["binary", "continuous"]:
        for model in config["model"][model_type]["models"]:
            config["model"][model_type]["hyperparameters"].setdefault(model, DEFAULT_HYPERPARAMETERS.get(model,{}))
    return config


#(for testing)...
if __name__ == "__main__":
    config = load_config()
    pprint.pprint(config, sort_dicts=False)
