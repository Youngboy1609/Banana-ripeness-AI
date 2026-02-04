import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def analyze_errors(preds_path: Path, output_dir: Path, top_k: int = 10):
    df = pd.read_csv(preds_path)
    
    # Filter for False Negatives in defects (True=defective, Pred=good)
    # y_defect_true is binary (1 for defective), y_defect_pred is binary (0 for good)
    fn_df = df[(df['y_defect_true'] == 1) & (df['y_defect_pred'] == 0)]
    
    print(f"Total False Negatives (Defective misclassified as Good): {len(fn_df)}")
    
    if len(fn_df) == 0:
        print("No errors found to analyze.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by defect confidence (logits or probability if available)
    # In this project, preds.csv usually has these columns
    if 'defect_prob' in fn_df.columns:
        fn_df = fn_df.sort_values(by='defect_prob', ascending=False) # Most "confident" errors first? 
        # Actually most "good" looking errors have low defect_prob
        fn_df = fn_df.sort_values(by='defect_prob', ascending=True)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, (idx, row) in enumerate(fn_df.head(top_k).iterrows()):
        img_path = Path(row['path'])
        if not img_path.is_absolute():
            img_path = ROOT / img_path
            
        if img_path.exists():
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f"Prob: {row.get('defect_prob', 'N/A'):.4f}\nRipeness: {row.get('y_ripeness_true', 'N/A')}")
        else:
            axes[i].text(0.5, 0.5, "Image not found", ha='center')
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_dir / "defect_false_negatives.png")
    print(f"Visualization saved to {output_dir / 'defect_false_negatives.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", default="reports/metrics/preds.csv")
    parser.add_argument("--output", default="reports/figures/error_analysis")
    args = parser.parse_args()
    
    preds_file = ROOT / args.preds
    if not preds_file.exists():
        print(f"Error: {preds_file} does not exist. Run evaluation first.")
    else:
        analyze_errors(preds_file, ROOT / args.output)
