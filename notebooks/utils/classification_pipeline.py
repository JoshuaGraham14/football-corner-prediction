import os
import sys
# Set project root
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root) 

import pandas as pd

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime 

from notebooks.utils.config_utils import load_config
from notebooks.utils.data_utils import load_dataset, preprocess_data
from notebooks.utils.model_utils import initialise_model, grid_search, get_feature_importance, optimise_threshold
from notebooks.utils.plotting_utils import (
    plot_correlation, plot_point_biserial_correlation, plot_prediction_distributions,
    plot_roc_and_prc, plot_confusion_matrices, plot_dataset_split,
    plot_backtesting_all_models 
)
from notebooks.utils.pdf_utils import (
    create_markdown_report, update_markdown_with_model_details,
    convert_markdown_to_html, save_pdf_from_html, append_pdf_with_combined_backtesting_results 
)
from notebooks.backtesting.backtester import Backtester
from notebooks.model_registry.model_registry import ModelRegistry 
 
"""
*** Classification Pipeline ***
- Setup:
    - Loads the config
    - Loads the corresponding track dataset with the selected features
    - Initialises model registry
- Feature Selection: 
    - Plots correlation heat maps
- Data Splitting:
    - Exclude the last 500 rows for testing (or 250 for track 2)
    - Split remaining data into 80% train and 20% validation sets
    - Plot the train/val/test split
- Initialise the PDF report:

*******
- Model Training Loop:
    1) Initialise model from config yaml
    2) Apply MinMax scaling (only for models that require scaling).
    3) Perform grid search for hyperparameter tuning (if specified) and train model
    4) Predict on validation set and display feature importance
    5) Precision threshold optimisation (maintining 10% min threshold)
    6) Evaluate model on validation set using optimised threshold
    7) Plot ROC and Precision-Recall graphs
    8) Predict on the test set and evaluate
    9) Save the model.
    10) Save predictions (for backtesting).
    11) Plot prediction distibutions (Scatter graph and KDE).
    12) Backtesting called and run
    13) Calculate evaluation metrics for model registry.
    13) Add results to PDF report.
*******

- Results:
    - Plot merged backtesting graph
    - Add each results to model registry (unless duplicate pipeline)
    - Generate and save PDF report
"""
def run_classification_pipeline(config_file_str, show_output=False, generate_pdf=False):
    #Load the config
    config = load_config(config_file_str)

    #Load the dataset
    df = load_dataset(config)
    df, selected_features, constructed_features, target_variable = preprocess_data(df, config)

    #Setup
    track_num = config['track_num']
    models_to_train = config['model']['classification']['models']
    apply_calibration = config['apply_calibration']
    testset_size = int(config['backtesting']['testset_size'])

    #Initialise the model registry:
    results_registry = ModelRegistry(track_num=track_num, selected_features=selected_features, constructed_features=constructed_features, is_calibration_applied=apply_calibration)
    model_results_dict = {}

    backtesting_histories_dict = {}
    
    #Only train on selected and constructed features
    X = df[selected_features + constructed_features]
    y = df[target_variable]

    #Get num of 1s and 0s
    value_counts=y.value_counts()
    num_ones=value_counts.get(1, 0)
    num_zeros=value_counts.get(0, 0)
    total=value_counts.sum()
    implied_1_plus_betting_odds = round(1/(num_ones/total), 4)

    print("--- DATASET STATS ---")
    print(f"Number of 1's: {num_ones}")
    print(f"Number of 0's: {num_zeros}")
    print(f"Total: {total}")
    print(f"Implied 1+ betting odds = {implied_1_plus_betting_odds}")
    print("---------------------")

    feature_correlation_image_path = plot_correlation(df, selected_features, constructed_features, [target_variable], track_num=track_num, show_output=show_output)
    point_biserial_correlation_image_path = plot_point_biserial_correlation(df, selected_features, constructed_features, track_num=track_num, show_output=show_output)

    #Split data to exclude the last 500 rows for testing
    train_data = df.iloc[:-testset_size]
    test_data = df.iloc[-testset_size:]

    # Split data -> train & validation
    # Info: 'stratify' param ensures train/validation split maintains same proportion of class labels - important for our imbalanced datasets.
    X_train, X_val, y_train, y_val = train_test_split(
        train_data[selected_features + constructed_features],
        train_data[target_variable],
        test_size=0.2,
        random_state=42,
        stratify=train_data[target_variable]
    )

    # Create X and y for testset
    X_test = test_data[selected_features+constructed_features]
    y_test = test_data[target_variable]

    # MinMax Scaling for models that require scaling...
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled =scaler.transform(X_val)
    X_test_scaled =scaler.transform(X_test)

    #Plots the train/val/test split of the dataset
    bar_charts_image_path, chronological_image_path = plot_dataset_split(train_data, X_val, y_val, test_data, target_variable, track_num=track_num, show_output=show_output)

    # Initialise and begin the Markdown for the report
    if generate_pdf:
        markdown_content = create_markdown_report(config, track_num, bar_charts_image_path, chronological_image_path, feature_correlation_image_path, point_biserial_correlation_image_path, target_variable, selected_features, constructed_features, models_to_train)

    #********************************************************************************************************
    # START OF MODEL TRAINING LOOP
    #********************************************************************************************************
    # Train and evaluate each model from the config
    for model_name in models_to_train:
        print(f"\n-> Training {model_name}...")
        if model_name not in model_results_dict:
            model_results_dict[model_name] = {}
    
        # Get the hyperparameters for the model
        hyperparameters = config["model"]["classification"]["hyperparameters"].get(model_name, {})
        do_grid_search=config["model"]["classification"].get("grid_search", False)

        # --- STEP 1: Initialise the model
        model = initialise_model(model_name, hyperparameters)

        # --- STEP 2: Select Scaled or Unscaled Data ---
        if model_name in ["logistic_regression", "svc", "xgboost"]:
            X_train, X_val, X_test = X_train_scaled, X_val_scaled, X_test_scaled
        
        # --- STEP 3: Grid Search (optional) ---
        if do_grid_search:
            optimal_model = grid_search(model_name, model, X_train, y_train, show_output)
        else:
            model.fit(X_train, y_train)
            optimal_model = model

        # --- STEP 4: Predict on the validation set ---
        y_pred_val = optimal_model.predict_proba(X_val)[:, 1]

        #Display Feature Importance
        feature_importances = get_feature_importance(optimal_model, model_name, selected_features, constructed_features)
        if show_output:
            print("\n### Top 8 and Bottom 5 Feature Importance ###")
            print(feature_importances)

        # --- STEP 5: Precision-Recall Threshold Optimisation ---
        best_threshold, optimal_threshold = optimise_threshold(y_pred_val, y_val, show_output=show_output, min_recall=0.05)

        # Info: Uncomment the line below to see what happens when every game is bet on.
        # optimal_threshold=0

        # --- STEP 6: Model Evaluation ---
        y_pred_threshold =(y_pred_val >= optimal_threshold).astype(int)
        if show_output:
            print(f"\n### Classification Report (threshold={optimal_threshold}):\n") 
            print(classification_report(y_val, y_pred_threshold))
        classification_report_str_1 = classification_report(y_val, y_pred_threshold, output_dict=False)

        # --- STEP 7: Plot ROC and Precision-Recall curves ---
        fpr,tpr,_ =roc_curve(y_val, y_pred_val)
        roc_auc =roc_auc_score(y_val, y_pred_val)
        precision,recall,_ =precision_recall_curve(y_val, y_pred_val)
        pr_auc=auc(recall, precision)

        # --- STEP 8: Test on test set ---
        # Predict on the final test data
        y_probs_final =optimal_model.predict_proba(X_test)[:, 1]
        y_pred_final=(y_probs_final >=optimal_threshold).astype(int)

        #Evaluate on the test set (final simulation)
        if show_output:
            print("\n### Prediction on last 500 rows: ###")
            print(classification_report(y_test, y_pred_final))
        classification_report_str_2 = classification_report(y_test, y_pred_final, output_dict=False)

        classification_report_image_path = plot_confusion_matrices(y_val, y_pred_threshold, y_test, y_pred_final, model_name, track_num=track_num, show_output=show_output)
        roc_prc_image_path = plot_roc_and_prc(fpr, tpr, roc_auc, precision, recall, pr_auc,model_name,track_num=track_num, show_output=show_output)

        # --- STEP 9: Save Model ---
        model_dir = f"../models/track{track_num}" 
        if not os.path.exists(model_dir): # Ensure the directory exists
            os.makedirs(model_dir)
        # Save the model
        joblib.dump(optimal_model, os.path.join(model_dir, f"{model_name.replace(' ', '_').lower()}_model_track{track_num}.pkl"))
        if show_output:
            print(f"{model_name} model saved.")

        prediction_file = f"../data/predictions/track{track_num}/{model_name.replace(' ', '_').lower()}_predictions_track{track_num}.csv"
        # --- STEP 10: Save Predictions ---
        results_df = pd.DataFrame({
            'kaggle_id': test_data['id_odsp'],
            'model_predicted_binary': y_pred_final,
            'actual_result': y_test
        })
        results_df.to_csv(prediction_file, index=False)
        if show_output:
            print(f"Predictions saved for {model_name}.")

        # --- STEP 11: Plot Prediction Distibution Graph ---
        scatter_image_path = plot_prediction_distributions(y_probs_final, y_test, model_name, track_num=track_num, show_output=show_output, optimal_threshold=optimal_threshold)

        # --- STEP 12: Backtesting called and run ---
        if show_output:
            print(f"\n-> Running Backtest for {model_name}...")
        odds_file = config["paths"]["total_corner_odds"]
        backtester = Backtester(config, odds_file=odds_file, model_name=model_name, model_file=prediction_file, track_num=track_num)
        backtesting_image_path, backtesting_results_str_list, backtesting_results_dict, bankroll_history = backtester.run(show_output)
        backtesting_histories_dict[model_name] = bankroll_history

        # --- STEP 13: Calculate eval metrics for model registry ---
        model_results_dict[model_name]['precision_val'] = round(precision_score(y_val, y_pred_threshold), 3)
        model_results_dict[model_name]['recall_val'] = round(recall_score(y_val, y_pred_threshold), 3)
        model_results_dict[model_name]['f1_score_val'] = round(f1_score(y_val, y_pred_threshold), 3)
        model_results_dict[model_name]['accuracy_val'] = round(accuracy_score(y_val, y_pred_threshold), 3)

        model_results_dict[model_name]['precision_test'] = round(precision_score(y_test, y_pred_final), 3)
        model_results_dict[model_name]['recall_test'] = round(recall_score(y_test, y_pred_final), 3)
        model_results_dict[model_name]['f1_score_test'] = round(f1_score(y_test, y_pred_final), 3)
        model_results_dict[model_name]['accuracy_test'] = round(accuracy_score(y_test, y_pred_final), 3)

        # Add backtesting results to model_results_dict
        model_results_dict[model_name].update(backtesting_results_dict)

        # --- STEP 14: Update PDF Report. ---
        if generate_pdf:
            #Finally, update markdown with generated outputs...
            markdown_content = update_markdown_with_model_details(
                markdown_content,
                model_name,
                feature_importances,
                best_threshold,
                classification_report_str_1,
                classification_report_str_2,
                classification_report_image_path,
                roc_prc_image_path,
                scatter_image_path,
                backtesting_results_str_list,
                backtesting_image_path
            )

    #********************************************************************************************************
    # END OF MODEL TRAINING LOOP
    #********************************************************************************************************
    
    #Plot merged backtesting graph:
    backtesting_all_image_path = plot_backtesting_all_models(track_num, backtesting_histories_dict, initial_bankroll=int(config["backtesting"]["initial_bankroll"]), show_output=show_output)
    #Add it to pdf report...
    if generate_pdf:
        markdown_content = append_pdf_with_combined_backtesting_results(
            markdown_content,
            backtesting_all_image_path
        )   
    
    #Add results to model registry (ONLY IF PIPELINE IS NOT DUPLICATE):
    if not results_registry.is_duplicate_pipeline:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M") #get timestamped before loop so they all are the same
        for model_name, model_results in model_results_dict.items():
            results_registry.add_model_results(
                timestamp, 
                model_name,
                model_results,
                selected_features,
                constructed_features,
                apply_calibration
            )
        print("\n📄 Saved Model Registry Results")
    
    #Generate and save PDF report
    if generate_pdf:
        now = datetime.now()
        date_time_str = now.strftime("%d-%m-%Y_%H:%M")
        folder_date_str = now.strftime("%d-%m-%Y")
        
        # Convert Markdown to html and save as pdf to reports/model_reports/
        html_content = convert_markdown_to_html(markdown_content)
        
        # Ensure the directory exists
        model_report_dir = f"../reports/model_reports/{folder_date_str}"
        if not os.path.exists(model_report_dir): 
            os.makedirs(model_report_dir)
            
        save_pdf_from_html(html_content, f'{model_report_dir}/track{track_num}_{date_time_str}_{results_registry.pipeline_id}.pdf')
        print("📄 Saved PDF Report")
    
    print("✅ Finished Running Pipeline")