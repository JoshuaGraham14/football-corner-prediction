"""
## Utility Funcs for Model Training:

- `Initialise_model`
    - Initialises classifier model depending on input model type

- `Grid_search`
    - Performs GridSearchCV for hyperparameter tuning depending on input model

- `Get_feature_importance`
    - Prints the top 8 and bottom 5 features

- `Optimise_threshold`
    - Optimises threshold based on Precision-Recall
    - Our aim is to increase Precision-Recall (more importantly precision), since we want to increase the likelihood of winning a 1+ corners at 80min bet -> prediciting the number of 1's correctly as important, i.e. when we do place a bet, we make sure we have a high chance of winning.
    - Therefore, I performed threshold adjustament to try and maximise precision (but ensure recall is at least 10% to avoid precision=1)
"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import pandas as pd
import numpy as np

def initialise_model(model_name, hyperparameters):
    if model_name == "random_forest":
        return RandomForestClassifier(**hyperparameters, random_state=42, class_weight="balanced")
    elif model_name == "logistic_regression":
        return LogisticRegression(**hyperparameters, random_state=42, class_weight="balanced")
    elif model_name == "svc":
        return SVC(probability=True, **hyperparameters, random_state=42, class_weight="balanced")
    elif model_name == "xgboost":
        return xgb.XGBClassifier(**hyperparameters, random_state=42)
    else:
        raise ValueError(f"Model, {model_name}, is not supported.")
    
def grid_search(model_name, model, X_train, y_train, show_output=True):
    """
    Performs Grid Search for hyperparameter tuning depending on input model:
    """
    if model_name == "random_forest":
        param_grid = param_grid or {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2,4],
            'bootstrap': [True, False]
        }

    elif model_name == "logistic_regression":
        param_grid = param_grid or {
            'C': [0.1, 1, 10,100],
            'solver': ['liblinear', 'saga']
        }

    elif model_name == "svc":
        param_grid = param_grid or {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma':['scale', 'auto']
        }

    elif model_name == "xgboost":
        param_grid = param_grid or {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'subsample': [0.8,1.0]
        }

    #Score by precision:
    precision_scorer = make_scorer(precision_score, average="micro")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=precision_scorer, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    if show_output:
        print("Best Parameters:", grid_search.best_params_)
        print("Best Precision Score:", grid_search.best_score_)

    #Use best_estimator for predictions
    best_model = grid_search.best_estimator_
    return best_model

def get_feature_importance(model, model_name, selected_features, constructed_features):
    """
    Gets feature importance (depending on model) 
    Displays top 8 + bottom 5...
    """
    # Collect all features
    all_features = selected_features + constructed_features
    
    if model_name in ["random_forest", "xgboost"]:
        importance = model.feature_importances_
    elif model_name == "logistic_regression":
        importance = model.coef_[0]
    elif model_name == "svc":
        #For SVC, feature importance is only available for linear
        if hasattr(model, 'coef_'):
            importance = model.coef_[0]
        else:  #Else, if its non-linear... set all importance to zero:
            importance = np.zeros(len(all_features)) 
    else:
        raise ValueError(f"Invalid model input: {model_name}")
    
    # Display feature importance
    feature_importances = pd.DataFrame({
        'Feature': all_features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    top_features = feature_importances.head(8) # Get top 8
    bottom_features = feature_importances.tail(5) # Get bottom 5
    
    # Combine the top 8 and bottom 5 features
    combined_features = pd.concat([top_features, bottom_features])

    return combined_features

def optimise_threshold(y_pred_val, y_val, show_output=True):
    #--- Threshold Maximisation ---
    thresholds = np.linspace(0.5, 0.95, 20) #test thresholds from 0.5 to 0.95
    results = []
    for t in thresholds:
        y_pred_t =(y_pred_val>=t).astype(int)
        precision_t= precision_score(y_val, y_pred_t, zero_division=1)  
        recall_t= recall_score(y_val, y_pred_t)
        
        if recall_t>=0.1: #only take results where recall >= 0.1
            results.append((t, precision_t, recall_t))

    best_threshold = max(results, key=lambda x: x[1])
    optimal_threshold = round(best_threshold[0],3)

    if show_output:
        #results as table
        print("\n### Precision-Recall Tradeoff at Different Thresholds ###\n") 
        print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}") 
        print("-"*36)  
        for t, p, r in results:
            print(f"{t:<12.2f}{p:<12.4f}{r:<12.4f}")

        print("\n### Recommended Threshold for Maximum Precision ###")
        print(f"Optimal Threshold: {optimal_threshold:.2f}")
        print(f"Expected Precision: {best_threshold[1]:.4f}")
        print(f"Expected Recall: {best_threshold[2]:.4f}")

    return best_threshold, optimal_threshold