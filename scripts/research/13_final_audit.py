import pandas as pd
from pathlib import Path

def final_audit(csv_path: Path):
    df = pd.read_csv(csv_path)
    
    # 1. Check if 'Augmented' source is in Test
    aug_in_test = df[(df['source'].str.contains('Augmented', case=False)) & (df['split'] == 'test')]
    print(f"\nAugmented samples in Test set: {len(aug_in_test)}")
    if len(aug_in_test) > 0:
        print("WARNING: Leakage detected! Augmented samples should only be in Train.")
    else:
        print("PASS: No augmented samples in Test set.")
    
    # 2. Check source exclusivity
    print("\nSource distribution across splits:")
    ct = pd.crosstab(df['source'], df['split'])
    print(ct)
    
    # 3. Analyze confidence in preds.csv if available
    preds_path = Path("reports/metrics/preds.csv")
    if preds_path.exists():
        pdf = pd.read_csv(preds_path)
        print("\n--- Prediction Confidence Analysis (on Test/Val) ---")
        if 'ripeness_prob' in pdf.columns: # Assuming column name
            print(f"Mean Ripeness Confidence: {pdf['ripeness_prob'].mean():.4f}")
        if 'defect_prob' in pdf.columns:
            print(f"Mean Defect Confidence: {pdf['defect_prob'].mean():.4f}")
            
        # Check for 100% confidence - common in leaked datasets
        if 'ripeness_prob' in pdf.columns:
            perfect = (pdf['ripeness_prob'] > 0.9999).mean() * 100
            print(f"Samples with >99.99% ripeness confidence: {perfect:.2f}%")
            
    # 4. Small sample check
    print("\nClass support in Test:")
    print(df[df['split'] == 'test'].groupby(['label_ripeness', 'label_defect']).size())

if __name__ == "__main__":
    final_audit(Path("data/processed/metadata.csv"))
