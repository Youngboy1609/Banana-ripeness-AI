import pandas as pd
from pathlib import Path
import os

def audit_metadata(csv_path: Path):
    df = pd.read_csv(csv_path)
    print(f"Total records: {len(df)}")
    
    # 1. Split Distribution
    print("\n--- Split Distribution ---")
    print(df['split'].value_counts())
    
    # 3. Source Overlap Check
    print("\n--- Source vs Split (Full) ---")
    ct = pd.crosstab(df['source'], df['split'])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(ct)
    
    # 4. Check for potential name-based leakage
    # If the filename (excluding source prefix) is the same across splits
    df['filename'] = df['path'].apply(lambda x: Path(x).name)
    # The processed filenames are like "{source_name}_{idx:06d}.jpg"
    # This renaming in prepare.py (line 944) actually HUDES the original filename.
    # This means we CANNOT audit leakage using filenames from metadata.csv!
    
    # Let's check if the 'group_id' logic in prepare.py was successful.
    # We can't see the group_id in metadata.csv though... it's not exported!
    
    # 5. Check if 'Augmented' source leaked into Test
    print("\n--- Augmented Source Distribution ---")
    aug_sources = [s for s in df['source'].unique() if 'Augmented' in s]
    for s in aug_sources:
        print(f"\nSource: {s}")
        print(df[df['source'] == s]['split'].value_counts())

    # 5. Overfitting check: Are there very few images in some categories?
    print("\n--- Class Counts in Test ---")
    test_df = df[df['split'] == 'test']
    print(test_df.groupby(['label_ripeness', 'label_defect']).size())

if __name__ == "__main__":
    audit_metadata(Path("data/processed/metadata.csv"))
