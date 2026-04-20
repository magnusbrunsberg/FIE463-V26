import json
import pandas as pd

notebook_path = 'solution.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell that generates master_results_pt6? No, the notebook is not "executed" in this script context. We can't see the output variables unless they are saved in the cell outputs!
# Let's read the cell outputs of the leaderboard.

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "output_df_cls.style.format" in source or "Best Neural Network" in source:
            outputs = cell.get('outputs', [])
            for out in outputs:
                if out['output_type'] == 'stream':
                    print("Stream Output:")
                    print("".join(out.get('text', [])))
                elif out['output_type'] == 'execute_result' or out['output_type'] == 'display_data':
                    data = out.get('data', {})
                    if 'text/plain' in data:
                        print("Text Output:")
                        print("".join(data['text/plain']))
                    if 'text/html' in data:
                        print("HTML Output found (not printing full HTML)")

