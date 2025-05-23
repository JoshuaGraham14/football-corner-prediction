import base64
from weasyprint import HTML
import markdown
from datetime import datetime

STYLE_STR = """
<style>
body {{
    font-family: 'Times New Roman', Times, serif;
    line-height: 1.6;
    color: #333;
    margin:0;
    padding:0;
}}
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Georgia', serif;
    color: #2c3e50;
    margin-bottom: 0.5em;
}}
h1{{
    font-size: 2.4em;
    border-bottom: 2px solid #2c3e50;
    padding-bottom: 0.3em;

}}
h2{{
    font-size: 1.4em;
    border-bottom: 1px solid #2c3e50;
    padding-bottom: 0.2em;
}}

p, pre,table {{
    font-family: 'Courier New',Courier, monospace;
    font-size: 0.8em;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 0.5em 0;
}}
th,td {{
    border: 1px solid #ddd;
    padding:4px;
    padding-left: 8px;
    text-align: left;
    font-size: 0.7em;
}}
th{{
    background-color: #f2f2f2;
}}
tr:nth-child(even) {{
    background-color:#f9f9f9;
}}

ul, ol {{
    margin: 0.5em 0;
    padding-left: 1.5em;
}}
li{{
    margin: 0.5em 0;
    font-family: 'Courier New', Courier, monospace;
    font-size:0.8em;
}}

.bold {{
    font-weight: bold;
}}
.italic {{
    font-style:italic;
}}
.bold-italic {{
    font-weight: bold;
    font-style:italic;
}}
.monospace {{
    font-family:'Courier New', Courier, monospace;
}}
</style>
"""

def create_markdown_report(config, track_num, dataset_split_bar_charts_image_path, dataset_split_chronological_image_path, feature_correlation_image_path, point_biserial_correlation_image_path, target_variable, selected_features, constructed_features, models_to_train):
    """
    Initialises a 'markdown' string for pdf report saving. 

    This function is responsible for adding the setup info to the pdf. This includes:
     - Track num
     - Target variable 
     - Features
     - Models
     - Feature Correlation
     - Dataset split
    """
    #Get date for pdf title...
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d, %H:%M")

    with open(feature_correlation_image_path, "rb") as image_file:
        encoded_feature_correlation_image = base64.b64encode(image_file.read()).decode('utf-8')
    with open(point_biserial_correlation_image_path, "rb") as image_file:
        encoded_point_biserial_correlation_image = base64.b64encode(image_file.read()).decode('utf-8')
    with open(dataset_split_bar_charts_image_path,"rb") as image_file:
        encoded_dataset_split_bar_charts_image =base64.b64encode(image_file.read()).decode('utf-8')
    with open(dataset_split_chronological_image_path,"rb") as image_file:
        encoded_dataset_split_chronological_image =base64.b64encode(image_file.read()).decode('utf-8')
    
    #Markdown content... starting with style sheet:
    markdown_content = STYLE_STR

    #Create cover page:
    markdown_content += """

#### {}
# Model Training Report: Track {}

## Configuration Details
#### Target Variable
{}

#### Selected Features
{}

#### Constructed Features
{}

#### Models Trained (& Hyperparameters)
{}

<div style="page-break-after:always;"></div>

#### Feature Correlation
<img src="data:image/png;base64,{}" style="max-width:100%; height:auto;">

#### Point-Biserial Correlation
<img src="data:image/png;base64,{}" style="max-width:100%; height:auto;">

<div style="page-break-after:always;"></div>

#### Dataset Train/Validation/Test Split Visualisation
<img src="data:image/png;base64,{}" style="max-width:100%; height:auto;">
<img src="data:image/png;base64,{}" style="max-width:100%; height:auto;">

<div style="page-break-after:always;"></div>
    """.format(
        date_time_str,
        track_num,
        target_variable,
        ", ".join(selected_features),
        ", ".join(constructed_features),
        "\n".join([f"- **{model}**: {config['model']['classification']['hyperparameters'].get(model,{})}" for model in models_to_train]),
        encoded_feature_correlation_image,
        encoded_point_biserial_correlation_image,
        encoded_dataset_split_bar_charts_image,
        encoded_dataset_split_chronological_image
    )

    return markdown_content

def update_markdown_with_model_details(markdown_content, model_name, feature_importances, best_threshold, classification_report_str_1, classification_report_str_2, classification_report_image_path, roc_prc_image_path, scatter_image_path, backtesting_results_list, backtesting_image_path):
    """
    This function is responsible for appending to the markdown string the results from each model pipeline run. This includes:
     - Feature importance
     - Confusion matrices
     - Optimal Threshold
     - ROC and Precision-Recall Curves
     - Backtesting results
    """
   
    # Convert image paths to base64:
    with open(classification_report_image_path,"rb") as image_file:
        encoded_classification_report_image =base64.b64encode(image_file.read()).decode('utf-8')
    with open(roc_prc_image_path,"rb") as image_file:
        encoded_roc_prc_image =base64.b64encode(image_file.read()).decode('utf-8')
    with open(scatter_image_path,"rb") as image_file:
        encoded_scatter_image =base64.b64encode(image_file.read()).decode('utf-8')
    with open(backtesting_image_path,"rb") as image_file:
        encoded_backtesting_image =base64.b64encode(image_file.read()).decode('utf-8')

    # Add feature importance table in Markdown format
    markdown_content += f"""
## {model_name} Model

#### Feature Importance

<table>
    <tr>
        <th>Top 8 Features</th>
        <th>Importance</th>
        <th>Bottom 5 Features</th>
        <th>Importance</th>
    </tr>
"""
    # Add top 8 features
    iterate_length = min(8, len(feature_importances))
    for i in range(iterate_length):
        feature = feature_importances.iloc[i]
        markdown_content += f"<tr><td>{feature.Feature}</td><td>{feature.Importance:.5f}</td>"
        if i < 5:
            bottom_feature = feature_importances.iloc[-(i+1)]
            markdown_content += f"<td>{bottom_feature.Feature}</td><td>{bottom_feature.Importance:.5f}</td></tr>"
        else:
            markdown_content += "<td></td><td></td></tr>"

    markdown_content += "</table>"

    # Add classification report
    markdown_content += f"""
    
<hr style="border:0.2px solid #2c3e50; margin:20px 0;">

#### Optimal Threshold
- **Optimal threshold**: {best_threshold[0]:.2f}
- **Expected Precision**: {best_threshold[1]:.3f}
- **Expected Recall**: {best_threshold[2]:.3f}

<hr style="border:0.2px solid #2c3e50; margin:20px 0;">

#### Classification Reports
<img src="data:image/png;base64,{encoded_classification_report_image}" style="max-width:100%; height:auto;">

<div style="page-break-after:always;"></div>
<hr style="border:0.2px solid #2c3e50; margin:20px 0;">
##### Classification Report (*Validation Set*) 

{classification_report_str_1}

##### Classification Report (*Test Set*)

{classification_report_str_2}

<hr style="border:0.2px solid #2c3e50; margin:20px 0;">

#### ROC and Precision-Recall Curves
<img src="data:image/png;base64,{encoded_roc_prc_image}" style="max-width: 100%;height:auto;">
<hr style="border:0.2px solid #2c3e50; margin:20px 0;">

<div style="page-break-after:always;"></div>

<hr style="border:0.2px solid #2c3e50; margin:20px 0;">
#### Prediction Distribution Graphs
<img src="data:image/png;base64,{encoded_scatter_image}" style="max-width: 100%;height:auto;">

<div style="page-break-after:always;"></div>
<hr style="border:0.2px solid #2c3e50; margin:20px 0;">
"""
    
    markdown_content += """
#### Backtesting Results
{}
<img src="data:image/png;base64,{}" style="max-width: 100%;height:auto;">

<div style="page-break-after:always;"></div>
    """.format(
        "<br>".join(result for result in backtesting_results_list),
        encoded_backtesting_image
    )

    return markdown_content

def append_pdf_with_combined_backtesting_results(markdown_content, backtesting_all_image_path):
    """
    This function is responsible for appending to the markdown string the results of the combined backtesting results. 
    """
    # Convert image paths to base64:
    with open(backtesting_all_image_path,"rb") as image_file:
        encoded_backtesting_all_image =base64.b64encode(image_file.read()).decode('utf-8')

    # Add feature importance table in Markdown format
    markdown_content += """
#### Backtesting Results Summary
<img src="data:image/png;base64,{}" style="max-width: 100%;height:auto;">

<div style="page-break-after:always;"></div>
    """.format(
        encoded_backtesting_all_image
    )

    return markdown_content

def convert_markdown_to_html(markdown_content):
    """
    Converts the markdown string into an HTML file.
    """
    html_content = markdown.markdown(markdown_content)
    return html_content

def save_pdf_from_html(html_content, output_pdf):
    """
    Converts the HTML content into a PDF file and saves it to the specified location.
    """
    HTML(string=html_content).write_pdf(output_pdf)