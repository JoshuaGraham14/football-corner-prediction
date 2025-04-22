import os
import pandas as pd
import hashlib

class ModelRegistry:
    def __init__(self, track_num, selected_features, constructed_features, is_calibration_applied):
        os.makedirs("../reports/model_registry", exist_ok=True)
        self.model_registry_file = f"../reports/model_registry/track{track_num}_results_22Apr.csv"
        
        #create the model_registry file if it doesn't exist...
        if not os.path.exists(self.model_registry_file):
            columns = [
                'pipeline_run_id', 'timestamp', 'model_name', 
                'selected_features', 'constructed_features', 'is_calibration_applied',
                'precision_val', 'recall_val', 'f1_score_val', 'accuracy_val', 
                'precision_test', 'recall_test', 'f1_score_test', 'accuracy_test',
                'total_profit', 'bets_placed', 'win_rate', 'roi', 'edge_over_bookies' 
            ]
            pd.DataFrame(columns=columns).to_csv(self.model_registry_file, index=False)
            print(f"Created new performance model registry file: {self.model_registry_file}")

        self.track_num = track_num
        self.is_duplicate_pipeline = False
        self.pipeline_run_id = self.generate_pipeline_id(selected_features, constructed_features, is_calibration_applied)

        # Check if this configuration already exists in the model registry
        model_registry_df=pd.read_csv(self.model_registry_file)
        existing_entry = model_registry_df[(model_registry_df['pipeline_run_id']==self.pipeline_run_id)]
        # If entry exists and has the same timestamp, don't duplicate
        if not existing_entry.empty:
            print(f"Duplicate pipeline found (pipeline_id = {self.pipeline_run_id}). Skipping model registry save...\n")
            self.is_duplicate_pipeline = True
        else:
            print(f"New pipeline created: (pipeline_id = {self.pipeline_run_id})\n")

    def generate_pipeline_id(self, selected_features, constructed_features, is_calibration_applied):
        """Generates an ID in a deterministic manner: based on config settings used"""

        #sort alphabetically... for consistency
        sorted_selected = sorted(selected_features)
        sorted_constructed = sorted(constructed_features)

        #Create string representation of config
        config_string = f"track{self.track_num}_selected:{','.join(sorted_selected)}_constructed:{','.join(sorted_constructed)}_calibration:{is_calibration_applied}"
        
        #Hash the string:
        hash_object = hashlib.md5(config_string.encode())
        return hash_object.hexdigest()[:8]
    
    def add_model_results(self, timestamp, model_name, model_results, selected_features, constructed_features, is_calibration_applied):
        """Add a model's results to the model registry under the specified pipeline run"""
        model_registry_df=pd.read_csv(self.model_registry_file)

        new_entry = {
            'pipeline_run_id': self.pipeline_run_id,
            'timestamp': timestamp,
            'model_name': model_name,
            'selected_features': ", ".join(selected_features),
            'constructed_features': ", ".join(constructed_features),
            'is_calibration_applied': is_calibration_applied,
            'precision_val': model_results.get('precision_val', 0),
            'recall_val': model_results.get('recall_val', 0),
            'f1_score_val': model_results.get('f1_score_val', 0),
            'accuracy_val': model_results.get('accuracy_val', 0),
            'precision_test': model_results.get('precision_test', 0),
            'recall_test': model_results.get('recall_test', 0),
            'f1_score_test': model_results.get('f1_score_test', 0),
            'accuracy_test': model_results.get('accuracy_test', 0),
            'total_profit': model_results.get('total_profit', 0),
            'bets_placed': model_results.get('num_bets', 0),
            'win_rate': model_results.get('win_rate', 0),
            'roi': model_results.get('roi', 0),
            'edge_over_bookies': model_results.get('edge', 0)
        }
        
        #Append to model_registry:
        model_registry_df=pd.concat([model_registry_df, pd.DataFrame([new_entry])], ignore_index=True)
        model_registry_df.to_csv(self.model_registry_file, index=False)
    