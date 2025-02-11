import os
import pandas as pd

def merge_csv_files(prefix, input_folder="results/"):
    output_filename = os.path.join(input_folder, f"{prefix}_final_output.csv")
    
    csv_files = [f for f in os.listdir(input_folder) if f.startswith(prefix) and f.endswith(".csv")]
    
    if not csv_files:
        print(f"No files found for prefix: {prefix}")
        return
    
    df_list = [pd.read_csv(os.path.join(input_folder, file)) for file in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    
    if prefix == 'batch':
        merged_df = merged_df.sort_values(by='batch_size')
    else:
        merged_df = merged_df.sort_values(by='prompt_length')
    
    merged_df.to_csv(output_filename, index=False)
    print(f"Merged {len(csv_files)} files into {output_filename}")

if __name__ == "__main__":
    for prefix in ["batch", "input"]:
        merge_csv_files(prefix)