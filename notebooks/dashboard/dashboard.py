import os
import pandas as pd
import uuid

class ModelDashboard:
    def __init__(self, track_num=1):
        os.makedirs("../reports/dashboard", exist_ok=True)
        self.dashboard_file = f"../reports/dashboard/track{track_num}.csv"
        
        #create the dashboard file if it doesn't exist...
        if not os.path.exists(self.dashboard_file):
            columns = [
                'pipeline_run_id', 'timestamp', 'model_name', 
                'selected_features', 'constructed_features', 'is_calibration_applied',
                'precision_val', 'recall_val', 'f1_score_val', 'accuracy_val', 
                'precision_test', 'recall_test', 'f1_score_test', 'accuracy_test',
                'total_profit', 'bets_placed', 'win_rate', 'roi', 'edge_over_bookies' 
            ]
            pd.DataFrame(columns=columns).to_csv(self.dashboard_file, index=False)
            print(f"Created new performance dashboard file: {self.dashboard_file}")

        self.pipeline_run_id = str(uuid.uuid4())[:8]
    
    def add_model_results(self, timestamp, model_name, model_results, selected_features, constructed_features, is_calibration_applied):
        """Add a model's results to the dashboard under the specified pipeline run"""

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
        
        #Append to dashboard:
        dashboard_df=pd.read_csv(self.dashboard_file)
        dashboard_df=pd.concat([dashboard_df, pd.DataFrame([new_entry])], ignore_index=True)
        dashboard_df.to_csv(self.dashboard_file, index=False)
        print(f"Added {model_name} results to the dashboard (pipeline run: {self.pipeline_run_id})")
    