import pandas as pd
import os
from pathlib import Path

def load_data(data_dir='data'):
    """
    Load and process data files from the data directory
    
    Args:
        data_dir (str): Path to data directory, defaults to 'data'
        
    Returns:
        pandas.DataFrame: Combined and shuffled dataset with 'text' and 'score' columns
    """
    # Create Path object
    data_path = Path(data_dir)
    
    # Check if directory exists
    if not data_path.exists():
        raise FileNotFoundError(f"Directory '{data_dir}' not found")
    
    # Initialize list to store all data
    all_data = []
    
    # Process each text file
    for file in data_path.glob('*_labelled.txt'):
        try:
            # Read file content
            df = pd.read_csv(file, sep='\t', header=None, names=['text', 'score'])
            print(f"Successfully processed {file.name}")
            all_data.append(df)
            
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
    
    # Combine all dataframes
    if not all_data:
        raise ValueError("No valid data files found")
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Shuffle the combined dataset
    shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return shuffled_df

if __name__ == "__main__":
    # Process data
    processed_data = load_data()
    
    # Save to CSV file
    output_file = 'data.csv'
    processed_data.to_csv(output_file, index=False)
    print(f"\nData saved to {output_file}")
    
    # Print summary of processed data
    print("\nData Processing Summary:")
    print(f"Total samples: {len(processed_data)}")
    print(f"Positive samples: {sum(processed_data['score'] == 1)}")
    print(f"Negative samples: {sum(processed_data['score'] == 0)}")
    print("\nFirst few samples:")
    print(processed_data.head())
