import pandas as pd
import argparse
import os

def count_labels(csv_file):
    """
    Count the number of ten-second segments for each label in a CSV file
    
    Args:
        csv_file (str): CSV file path
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Check if label column exists
        if 'label' not in df.columns:
            print(f"Error: 'label' column not found in CSV file")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Count segments for each label
        label_counts = df['label'].value_counts().sort_index()
        
        # Print results
        print(f"File: {csv_file}")
        print(f"Total segments: {len(df)}")
        print("\nSegment count for each label:")
        print("-" * 40)
        
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"Label {label}: {count} segments ({percentage:.2f}%)")
        
    except FileNotFoundError:
        print(f"Error: File not found {csv_file}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Count ten-second segments for each label in a CSV file')
    parser.add_argument('csv_file', help='CSV file path')
    
    args = parser.parse_args()
    
    count_labels(args.csv_file)

if __name__ == '__main__':
    # If no command line arguments, use default file
    import sys
    if len(sys.argv) == 1:
        # Default to merged_labels.csv in current directory
        default_file = 'merged_labels.csv'
        if os.path.exists(default_file):
            print(f"Using default file: {default_file}")
            count_labels(default_file)
        else:
            print("Please provide CSV file path as argument")
            print("Usage: python count_labels.py <csv_file>")
    else:
        main() 