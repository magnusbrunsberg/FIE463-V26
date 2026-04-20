import json

notebook_path = 'solution.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Replace grouping lines
            if "ts_act = df_eval.groupby('date')[target].mean()" in line:
                line = line.replace("df_eval.groupby('date')[target].mean()", "df_eval.set_index('date').resample('ME')[target].mean()")
            if "ts_pred = df_eval.groupby('date')['y_predicted'].mean()" in line:
                line = line.replace("df_eval.groupby('date')['y_predicted'].mean()", "df_eval.set_index('date').resample('ME')['y_predicted'].mean()")
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
