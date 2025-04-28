"""
## Plotting Utils 

- `Plot_scatter` - Plot scatter graph to visualise our spread of predictions & how the threshold binarises them. 

- `Plot_roc_and_prc` (calls `plot_roc_curve` and `plot_precision_recall_curve`)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import pointbiserialr
from sklearn.calibration import calibration_curve

def plot_correlation(df_selected, selected_features, constructed_features, target_variables, show_output=True):
    #Corelation matrix:
    correlation_matrix = df_selected[selected_features + constructed_features + target_variables].corr()
    correlation_with_target = correlation_matrix[target_variables].drop(target_variables, axis=0).abs()

    # Plot correlation heatmap:
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_with_target, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.xlabel("Target Variable")
    plt.ylabel("Features")
    plt.title("Correlation between Features and Target variable")
    plt.tight_layout()

    # Save graph as an image
    feature_correlation_image_path = f"../reports/images/feature_correlation.png"
    plt.savefig(feature_correlation_image_path)
    
    if show_output:
        plt.show()
    else:
        plt.close()

    return feature_correlation_image_path

def plot_calibration_curve(y_true, y_prob, model_name, show_output=False):
    """Plots a calibration curve for the inputted model"""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2,label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--',label='Perfectly calibrated')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability in each bin')
    plt.title(f'Calibration Curve -{model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save graph as an image
    image_path = f"../reports/images/{model_name.replace(' ', '_').lower()}_calibration.png"
    plt.savefig(image_path)
    
    if show_output:
        plt.show()
    else:
        plt.close()

def plot_prediction_distributions(y_probs_final, y_test_final, model_name, show_output=True, optimal_threshold=0.5):
    #Split y_probs_final by target = 0 or 1...
    y_probs_class0 = y_probs_final[y_test_final == 0]
    y_probs_class1 = y_probs_final[y_test_final == 1]
    
    # # Calculate mean probabilities for each class
    # mean_prob_class0 = np.mean(y_probs_class0)
    # mean_prob_class1 = np.mean(y_probs_class1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Scatter plot: target vs predicted probabilities
    ax1.scatter(range(len(y_probs_final)), y_probs_final, c=y_test_final, cmap='coolwarm', alpha=0.7, label='Predicted Probability')

    ax1.axhline(optimal_threshold, color='grey', linestyle='dashed', linewidth=2, label=f'Threshold = {optimal_threshold:.2f}')
    
    # plot mean probability for each class:
    # ax1.axhline(mean_prob_class0, color='blue', linestyle='-', linewidth=1.5, 
    #             label=f'Mean (Class 0) = {mean_prob_class0:.2f}')
    # ax1.axhline(mean_prob_class1, color='red', linestyle='-', linewidth=1.5, 
    #             label=f'Mean (Class 1) = {mean_prob_class1:.2f}')

    class_0 =mlines.Line2D([], [],color='blue',marker='o',linestyle='None', markersize=8, alpha=0.7, label='Target: No 1+ Corners')
    class_1=mlines.Line2D([], [],color='red', marker='o',linestyle='None',markersize=8,alpha=0.7, label='Target: 1+ Corners')
    ax1.set_xlabel('Match Index')
    ax1.set_ylabel('Predicted probability of 1+ corners')
    ax1.set_title(f'Predicted Probability vs Actual Target ({model_name})')
    ax1.legend(handles=[class_0, class_1, 
                        plt.Line2D([], [], color='grey',linestyle='dashed',linewidth=2, label=f'Threshold = {optimal_threshold:.2f}'),
                        # plt.Line2D([], [], color='blue', linestyle='-', linewidth=1.5, label=f'Mean (Class 0) = {mean_prob_class0:.3f}'),
                        # plt.Line2D([], [], color='red', linestyle='-', linewidth=1.5, label=f'Mean (Class 1) = {mean_prob_class1:.3f}')
        ])
    
    #KDE plot: density of probabilities, split by actual true target
    sns.kdeplot(y_probs_class0, ax=ax2, fill=True,color='blue', alpha=0.5,label='Target: No 1+ Corners')
    sns.kdeplot(y_probs_class1, ax=ax2,fill=True, color='red', alpha=0.5,label='Target: 1+ Corners')
    ax2.axvline(optimal_threshold, color='grey', linestyle='dashed', linewidth=2, label=f'Threshold = {optimal_threshold:.2f}')
    ax2.set_xlabel('Predicted probability of 1+ corners')
    ax2.set_ylabel('Density')
    ax2.set_title(f'KDE Distribution of Predicted Probabilities by Class ({model_name})')
    ax2.legend()
    plt.tight_layout()


    # Save graph as an image
    image_path = f"../reports/images/{model_name.replace(' ', '_').lower()}_prediction_distributions.png"
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
    
def plot_precision_recall_curve(precision, recall, pr_auc, model_name, ax):
    ax.plot(recall, precision, color='b', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve - {model_name}')
    ax.legend(loc="lower left")

def plot_roc_and_prc(fpr, tpr, roc_auc, precision, recall, pr_auc,model_name, show_output=True):
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
    sns.heatmap(corr_df[['P-value']],annot=True, cmap='coolwarm_r',ax=axes[1])
    axes[1].set_title('P-value Heatmap')
    sns.heatmap(corr_df[['Combined Score']],annot=True, cmap='coolwarm',ax=axes[2])
    axes[2].set_title('Combined Score Heatmap')
    plt.tight_layout()

    # Save graph as an image
    point_biserial_correlation_image_path = f"../reports/images/point_biserial_correlation.png"
    plt.savefig(point_biserial_correlation_image_path)
    
    plt.show()

    return point_biserial_correlation_image_path

def plot_dataset_split(train_data, X_val, y_val, test_data, target_variable, show_output=True, track_num=1):
    """
    Create visualisation of train/validation/test split
    """
    BAR_COLOURS=['dodgerblue', 'seagreen', 'indianred']

    #Since 80/20 train data split is done in the main func, we have to use the X_val and y_val variables
    # --- Calculate each segments size --- 
    train_size = len(train_data)-len(X_val)
    val_size = len(X_val)
    test_size = len(test_data)
    total_size = train_size + val_size +test_size
    
    # --- Calculate target distribution in each segement ---
    train_positive = train_data[target_variable].sum()-y_val.sum()
    train_negative = train_size-train_positive
    
    val_positive = y_val.sum()
    val_negative = val_size-val_positive
    
    test_positive = test_data[target_variable].sum()
    test_negative = test_size - test_positive
    
    # Plot two displays:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Plot 1: Dataset split sizes ---
    sizes = [train_size,val_size, test_size]
    labels=[f'Train\n({train_size} rows)', f'Validation\n({val_size} rows)', f'Test\n({test_size} rows)']
    ax1.bar(labels, sizes, color=BAR_COLOURS)
    ax1.set_title(f'Train/Validation/Test Split - Distribution (Track {track_num})',fontsize=14)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    # Adds percentages above bars
    for i, v in enumerate(sizes):
        ax1.text(i, v+0.2, f'{v/total_size:.1%}', ha='center',fontsize=10)
    
    # --- Plot 2: Target distribution in each dataset split ---
    width= 0.3 
    x=np.arange(3)
    positives =[train_positive, val_positive, test_positive]
    negatives =[train_negative, val_negative, test_negative]
    ax2.bar(x - width/2, positives, width,label=f'{target_variable}=1', color='darkorange')
    ax2.bar(x + width/2, negatives, width, label=f'{target_variable}=0',color='slategray')
     
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Train', 'Validation', 'Test'])
    ax2.set_title(f'Train/Validation/Test Split - Target Variable Distribution (Track {track_num})', fontsize=14)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.legend()
    
    # Adds percentages above bars
    #positive target bars first 
    for i, v in enumerate(positives):
        total=positives[i]+negatives[i]
        percentage = v/total
        ax2.text(i - width/2, v+0.2, f'{percentage:.1%}', ha='center',fontsize=9)
    #then negative target bars first
    for i, v in enumerate(negatives): 
        total=positives[i]+negatives[i] 
        percentage = v/total
        ax2.text(i + width/2, v+0.2, f'{percentage:.1%}', ha='center',fontsize=9)

    # Save and display plot...
    plt.tight_layout()

    bar_charts_image_path = f"../reports/images/dataset_split/track{track_num}_barcharts.png"
    os.makedirs(os.path.dirname(bar_charts_image_path), exist_ok=True)
    plt.savefig(bar_charts_image_path, dpi=300, bbox_inches='tight')

    if show_output:
        plt.show()
    else:
        plt.close()

    #---------------------------------------
    
    # Plot 3: Chronological dataset split distribution (on new plot)
    fig2, ax2 = plt.subplots(figsize=(20, 3))
     
    # Create a horizontal bar with segments for each split
    labels=['Train', 'Validation', 'Test']
    sizes = [train_size, val_size, test_size] 
    
    #Draw 3 horizontal bars:
    leftPointer=0
    for i, (size, color, label) in enumerate(zip(sizes, BAR_COLOURS, labels)):
        ax2.barh(0, size,left=leftPointer, color=color, height=0.2,label=label)
        # Add text in the middle of each one...
        percentage = (size/total_size) * 100
        text_x = leftPointer + size/2
        ax2.text(text_x, 0, f"{label}\n{size} rows\n({percentage:.1f}%)", 
                ha='center',va='center', color='white',fontweight='bold')
        leftPointer += size    
    
    # Add distribution information as text below the bar...
    annotations = [
        f"Target=1: {train_positive/train_size:.1%} \nTarget=0: {train_negative/train_size:.1%}",
        f"Target=1: {val_positive/val_size:.1%} \nTarget=0: {val_negative/val_size:.1%}", 
        f"Target=1: {test_positive/test_size:.1%} \nTarget=0: {test_negative/test_size:.1%}" 
    ]
    leftPointer=0
    for i, (size, annotation) in enumerate(zip(sizes, annotations)):
        ax2.text(leftPointer + size/2, -0.2, annotation, ha='center', va='top')
        leftPointer += size
    
    # Format plot
    ax2.set_yticks([]) 
    ax2.set_xlim(0, total_size) 
    #Create x axis ticks...
    regular_ticks = [tick for tick in range(0, total_size-1000, 1000)]
    #... and special ticks for the split boundaries and total:
    special_ticks = [0, total_size]
    all_ticks = sorted(list(set(regular_ticks +special_ticks)))
    ax2.set_xticks(all_ticks)
    
    # Format tick labels:
    tick_labels = [str(tick) for tick in all_ticks]
    if total_size in all_ticks:
        total_index = all_ticks.index(total_size)
        tick_labels[total_index]=f'{total_size}' #add final total tick
    ax2.set_xticklabels(tick_labels)
    
    # vertical lines at split boundaries...
    ax2.axvline(x=train_size, color='gray',linestyle='--',alpha=0.5)
    ax2.axvline(x=train_size + val_size, color='gray',linestyle='--',alpha=0.5)
    
    ax2.set_xlabel('Game Index', fontsize=12)
    ax2.set_title(f'Train/Validation/Test Split - Sequential Visualisation (Track {track_num})', fontsize=14)
    
    #Add legend
    ax2.legend(loc='lower left', bbox_to_anchor=(0.01, -0.5), ncol=3)

    plt.tight_layout()

    chronological_image_path = f"../reports/images/dataset_split/track{track_num}_chronological.png"
    os.makedirs(os.path.dirname(chronological_image_path), exist_ok=True)
    plt.savefig(chronological_image_path, dpi=300, bbox_inches='tight')

    if show_output:
        plt.show()
    else:
        plt.close()

    return bar_charts_image_path, chronological_image_path
    
def plot_backtesting_all_models(histories_dict, initial_bankroll=1000, track_num=1, show_output=True):
    """
    Creates a comparison chart of multiple model backtesting simulations
    -> histories_dict (dict): keys = model names, values = bankroll history lists
    """
    plt.figure(figsize=(12,6))
    for label, history in histories_dict.items(): 
        plt.plot(history,label=label, linewidth=2)
 
    plt.axhline(y=initial_bankroll, color='gray',linestyle='--')
    plt.title("Backtesting Bankroll Growth – Track Comparison")
    plt.xlabel("Games Simulated")
    plt.ylabel("Bankroll (£)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout() 
     
    # Save plot as an image:
    image_path =f"../reports/images/backtesting/track{track_num}_all.png" 
    plt.savefig(image_path)
    
    if show_output:
        plt.show() 
    else:
        plt.close()

    return image_path 
