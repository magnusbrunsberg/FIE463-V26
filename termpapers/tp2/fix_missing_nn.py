import json

notebook_path = 'solution.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any("def calculate_binary_classification" in line for line in source) and ("Gradient Boosting" in "".join(source)):
            new_source = []
            for line in source:
                if "# Hard predictions" in line:
                    if not any("elif m_name == 'Neural Network':" in l for l in new_source):
                        inject = """        elif m_name == 'Neural Network':
            nn_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))
            ])
            nn_pipe.fit(X_train, y_train)
            fitted = nn_pipe
            best_param = "64x32 Nodes"

"""
                        new_source.extend([l + '\n' for l in inject.strip('\n').split('\n')])
                    new_source.append(line)
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Neural Network branch injected properly in the exact cell!")
