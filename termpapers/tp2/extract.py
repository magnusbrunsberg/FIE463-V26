import json
import os
import sys

def extract_notebook(filename, out_filename):
    with open(filename, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(out_filename, 'w', encoding='utf-8') as f:
        for cell in nb.get('cells', []):
            f.write(f"----- {cell['cell_type'].upper()} -----\n")
            source = "".join(cell.get('source', []))
            f.write(source + "\n\n")

extract_notebook('solution.ipynb', 'solution_text.txt')
extract_notebook('solution_2004.ipynb', 'solution_2004_text.txt')
print("Successfully extracted notebooks.")
