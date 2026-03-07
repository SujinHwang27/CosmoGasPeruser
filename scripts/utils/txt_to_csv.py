import re
import csv
import os

def convert_txt_to_csv(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    csv_data = []
    headers = []

    for line in lines:
        line = line.strip()
        # Clean up the box characters and extra whitespace
        text = line.replace('│', '').replace('├', '').replace('┤', '').replace('┌', '').replace('┐', '').replace('└', '').replace('┘', '').strip()
        
        if not text or text.startswith('Flux Matrix') or text.startswith('Rows ='):
            continue
            
        # Parse headers
        if text.startswith('Col 0'):
            # Split by multiple spaces, but handle the ellipsis
            parts = re.split(r'\s{2,}', text)
            headers = ["Row_ID"] + [p for p in parts if p != '...']
            continue

        # Parse data rows
        if text.startswith('Row'):
            # Look for "Row X" and then numbers
            row_match = re.match(r'Row\s+(\d+)\s+(.*)', text)
            if row_match:
                row_id = row_match.group(1)
                data_part = row_match.group(2)
                # Split the data part by spaces, ignore '...'
                values = [v for v in data_part.split() if v != '...']
                csv_data.append([row_id] + values)

    if csv_data:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if headers:
                writer.writerow(headers)
            writer.writerows(csv_data)
        print(f"Successfully converted {input_path} to {output_path}")
    else:
        print("No data found to convert.")

if __name__ == "__main__":
    # Convert value_print.txt
    convert_txt_to_csv("eda_plots/value_print.txt", "eda_plots/value_print.csv")
    
    # Convert tier1_summary_stats.txt
    # Since it's already comma-separated, a simple copy/rename is enough
    # But we can use the same logic or just a simple write if we want to be safe
    import shutil
    src = "eda_plots/tier1_summary_stats.txt"
    dst = "eda_plots/tier1_summary_stats.csv"
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Successfully converted {src} to {dst}")
