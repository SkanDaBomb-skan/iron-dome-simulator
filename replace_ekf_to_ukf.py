import json
import os

def fix_notebook(filepath):
    print(f"Fixing {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    changes = 0
    for cell in nb.get('cells', []):
        if not cell.get('source'):
            continue
            
        new_source = []
        for line in cell['source']:
            original = line
            
            # Replace case-sensitive mentions
            line = line.replace('EKF', 'UKF')
            line = line.replace('ekf', 'ukf')
            
            # Fix hardcoded ~697m to ~209m
            line = line.replace('~697m', '~209m')
            
            if line != original:
                changes += 1
            new_source.append(line)
            
        cell['source'] = new_source
        
        # Clear outputs for code cells to avoid stale text
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
            
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        
    print(f"  -> Made {changes} line changes in {filepath}")

notebooks = [
    'notebooks/phase2_prediction.ipynb',
    'notebooks/phase2_prediction_3d.ipynb'
]

for nb in notebooks:
    if os.path.exists(nb):
        fix_notebook(nb)
