import json

notebook_path = 'solution.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# First pass: add missing RandomForestRegressor import
for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "RandomForestClassifier" in source_str:
            if "RandomForestRegressor" not in source_str:
                new_source = []
                for line in cell['source']:
                    new_source.append(line)
                    if "from sklearn.ensemble import RandomForestClassifier" in line:
                        new_source.append("from sklearn.ensemble import RandomForestRegressor\n")
                cell['source'] = new_source
                break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("RandomForestRegressor import added successfully.")
