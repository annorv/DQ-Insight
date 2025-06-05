# export.py

import pandas as pd
import io

def export_results_to_csv(results):
    """Export the results of data quality checks to a CSV format."""
    export_data = {
        'Metric': [],
        'Results': [],
        'Recommendations': []
    }
    
    for metric, result in results.items():
        export_data['Metric'].append(metric)
        if 'data' in result and not result['data'].empty:
            export_data['Results'].append(result['data'].to_string())
        else:
            export_data['Results'].append("")
        export_data['Recommendations'].append(result['description'])
    
    results_df = pd.DataFrame(export_data)
    
    # Ensure all lists in the DataFrame are the same length
    max_length = max(len(export_data['Metric']), len(export_data['Results']), len(export_data['Recommendations']))
    for key in export_data:
        while len(export_data[key]) < max_length:
            export_data[key].append("")

    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    
    return csv_buffer.getvalue()
