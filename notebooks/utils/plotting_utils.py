"""
## Plotting Utils 

- `Plot_scatter` - Plot scatter graph to visualise our spread of predictions & how the threshold binarises them. 

- `Plot_roc_and_prc` (calls `plot_roc_curve` and `plot_precision_recall_curve`)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import pointbiserialr

def plot_correlation(df_selected, selected_features, constructed_features, target_variables, show_output=True):
    #Corelation matrix:
    correlation_matrix = df_selected[selected_features + constructed_features + target_variables].corr()
    correlation_with_target = correlation_matrix[target_variables].drop(target_variables, axis=0)

    # Plot correlation heatmap:
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_with_target, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.xlabel("Target Variables")
    plt.ylabel("Constructed Features")
    plt.title("Correlation between Constructed Features and Target Variables")
    plt.tight_layout()

    # Save graph as an image
    feature_correlation_image_path = f"../reports/images/feature_correlation.png"
    plt.savefig(feature_correlation_image_path)
    
    if show_output:
        plt.show()
    else:
        plt.close()

    return feature_correlation_image_path

def plot_scatter(y_probs_final, y_test_final, optimal_threshold, model_name, show_output=True):
    # Scatter plot: target vs predicted probabilities
    plt.figure(figsize=(12, 5))
    plt.scatter(range(len(y_probs_final)), y_probs_final,c=y_test_final,cmap='coolwarm',alpha=0.7,label='Predicted Probability')
    plt.axhline(optimal_threshold, color='red', linestyle='dashed',linewidth=2,label=f'Threshold = {optimal_threshold:.2f}')
    class_0 =mlines.Line2D([], [],color='blue',marker='o',linestyle='None', markersize=8, alpha=0.7, label='Target: No 1+ Corners')
    class_1=mlines.Line2D([], [],color='red', marker='o',linestyle='None',markersize=8,alpha=0.7, label='Target: 1+ Corners')
    plt.xlabel('Match Index')
    plt.ylabel('Predicted probability of 1+ corners')
    plt.title(f'{model_name} - Predicted Probability vs Actual Target')
    plt.legend(handles=[class_0, class_1,plt.Line2D([], [],color='grey',linestyle='dashed',linewidth=2,label='Threshold')])

    # Save graph as an image
    image_path = f"../reports/images/{model_name.replace(' ', '_').lower()}_scatter.png"
    plt.savefig(image_path)
    
    if show_output:
        plt.show()
    else:
        plt.close()

    return image_path #return path for use later in markdwon func

def plot_roc_curve(fpr, tpr, roc_auc, model_name, ax):
    ax.plot(fpr,tpr, color='b',lw=2, label=f'ROC curve (area ={roc_auc:.2f})')
    ax.plot([0,1], [0,1], color='gray', linestyle='--')
    ax.set_xlim([0.0,1.0])
    ax.set_ylim([0.0,1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc="lower right")
    
def plot_precision_recall_curve(precision, recall,pr_auc, model_name, ax):
    ax.plot(recall, precision, color='b', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve - {model_name}')
    ax.legend(loc="lower left")

def plot_roc_and_prc(fpr, tpr, roc_auc, precision,recall, pr_auc,model_name, show_output=True):
    #Plots ROC curve and Precision-recall side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  
    plot_roc_curve(fpr, tpr, roc_auc, model_name,ax1)
    plot_precision_recall_curve(precision, recall, pr_auc, model_name,ax2)
    plt.tight_layout()

    # Save graph as an image
    image_path = f"../reports/images/{model_name.replace(' ', '_').lower()}_roc_prc.png"
    plt.savefig(image_path)
    # plt.close()

    if show_output:
        plt.show()
    else:
        plt.close()

    return image_path

def plot_classification_report(optimal_model, X_val, y_val, X_test, y_test, model_name, show_output=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    #Plot for validation set:
    ConfusionMatrixDisplay.from_estimator(optimal_model, X_val, y_val,ax=axes[0],cmap='Blues',values_format='d')
    axes[0].set_title(f'{model_name} - Validation Set')

    #Plot for test set:
    ConfusionMatrixDisplay.from_estimator(optimal_model, X_test, y_test,ax=axes[1],cmap='Blues',values_format='d')
    axes[1].set_title(f'{model_name} - Test Set')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3) #line reduces plot whitespace borders
    # Save graph as an image
    image_path =f"../reports/images/{model_name.replace(' ', '_').lower()}_confusion_matrix.png"
    plt.savefig(image_path)
    # plt.close()
    
    if show_output:
        plt.show()
    else:
        plt.close()

    return image_path

def plot_point_biserial_correlation (df_selected, selected_features, constructed_features):
    #Calc Point-Biserial correlation and P-values:
    correlation_results = [
        (col, *pointbiserialr(df_selected[col], df_selected['target']))
        for col in df_selected[selected_features+constructed_features] if col !='target'
    ]
    corr_df = pd.DataFrame(correlation_results, columns=['Feature', 'Correlation','P-value'])
    corr_df.set_index('Feature', inplace=True)

    corr_df['Abs Correlation'] =abs(corr_df['Correlation'])
    corr_df['Combined Score'] =corr_df['Abs Correlation']*(1-corr_df['P-value'])

    #Print top 10 features
    print("--- TOP 10 FEATURES ---")
    print(corr_df['Combined Score'].nlargest(10))
    print("-----------------------")

    # Plot heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))
    sns.heatmap(corr_df[['Abs Correlation']],annot=True, cmap='coolwarm',ax=axes[0])
    axes[0].set_title('Point-Biserial Correlation Heatmap')
    sns.heatmap(corr_df[['P-value']],annot=True, cmap='coolwarm',ax=axes[1])
    axes[1].set_title('P-value Heatmap')
    sns.heatmap(corr_df[['Combined Score']],annot=True, cmap='coolwarm_r',ax=axes[2])
    axes[2].set_title('Combined Score Heatmap')
    plt.tight_layout()

    # Save graph as an image
    point_biserial_correlation_image_path = f"../reports/images/point_biserial_correlation.png"
    plt.savefig(point_biserial_correlation_image_path)
    
    plt.show()

    return point_biserial_correlation_image_path

   