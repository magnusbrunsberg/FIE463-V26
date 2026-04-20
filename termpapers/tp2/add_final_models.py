import json

notebook_path = 'solution.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# First pass: add imports
for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "from sklearn.ensemble import HistGradientBoostingClassifier" in source_str or "from sklearn.ensemble import RandomForestClassifier" in source_str:
            new_source = []
            for line in cell['source']:
                new_source.append(line)
                if "from sklearn.ensemble import RandomForestClassifier" in line:
                    if "MLPClassifier" not in source_str:
                        new_source.append("from sklearn.neural_network import MLPClassifier\n")
            cell['source'] = new_source
            break

# Update part 5 models (Add Random Forest)
for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any("GradientBoosting" in line for line in source) and any("LassoCV(alphas=" in line for line in source):
            if not any("RandomForestRegressor" in line for line in source):
                new_source = []
                for line in source:
                    if "'GradientBoosting': HistGradientBoostingRegressor" in line:
                        if line.rstrip().endswith(','):
                            new_source.append(line)
                        else:
                            new_source.append(line.rstrip('\n') + ',\n')
                        new_source.append("        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)\n")
                    else:
                        new_source.append(line)
                cell['source'] = new_source

# Update part 6 models (Add Neural Network)
for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any("class_models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']" in line for line in source):
            new_source = []
            for line in source:
                if "class_models =" in line:
                    new_source.append("    class_models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Neural Network']\n")
                elif "elif m_name == 'Gradient Boosting':" in line:
                    # Insert before Gradient Boosting or after it
                    new_source.append(line)
                elif "# Probability predictions" in line or "# Hard predictions" in line:
                    if not any("Neural Network" in l for l in new_source):
                        inject = """
        elif m_name == 'Neural Network':
            nn_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))
            ])
            nn_pipe.fit(X_train, y_train)
            fitted = nn_pipe
            best_param = "64x32 Nodes"
"""
                        # Find the place to inject before "Hard Predictions". 
                        new_source.extend([l + '\n' for l in inject.strip('\n').split('\n')])
                    new_source.append(line)
                else: 
                    new_source.append(line)
            cell['source'] = new_source

# Update part 6 plotting
for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "best_gb = max" in source_str and "fig, axes = plt.subplots" in source_str:
            
            rewritten = """for target in optimist_vars:
    target_models = [m for m in master_results_pt6 if m['Target'] == target]

    best_logit = max([m for m in target_models if m['Classifier'] == 'Logistic Regression'], key=lambda x: x['Test F1'])
    best_rf = max([m for m in target_models if m['Classifier'] == 'Random Forest'], key=lambda x: x['Test F1'])
    best_gb = max([m for m in target_models if m['Classifier'] == 'Gradient Boosting'], key=lambda x: x['Test F1'])
    best_nn = max([m for m in target_models if m['Classifier'] == 'Neural Network'], key=lambda x: x['Test F1'])

    print(f"\\n{target}")
    print(f"Best Logistic: {best_logit['Features']} | Test F1 = {best_logit['Test F1']:.4f}")
    print(f"Best Random Forest: {best_rf['Features']} | Test F1 = {best_rf['Test F1']:.4f}")
    print(f"Best Gradient Boosting: {best_gb['Features']} | Test F1 = {best_gb['Test F1']:.4f}")
    print(f"Best Neural Network: {best_nn['Features']} | Test F1 = {best_nn['Test F1']:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes = axes.flatten()

    plot_fractional_forecast(best_logit, ax=axes[0])
    axes[0].set_title(f"{target} | Logistic Regression\\n{best_logit['Features']} | Test F1 = {best_logit['Test F1']:.3f}", fontsize=11, fontweight='bold')

    plot_fractional_forecast(best_rf, ax=axes[1])
    axes[1].set_title(f"{target} | Random Forest\\n{best_rf['Features']} | Test F1 = {best_rf['Test F1']:.3f}", fontsize=11, fontweight='bold')
    
    plot_fractional_forecast(best_gb, ax=axes[2])
    axes[2].set_title(f"{target} | Gradient Boosting\\n{best_gb['Features']} | Test F1 = {best_gb['Test F1']:.3f}", fontsize=11, fontweight='bold')

    plot_fractional_forecast(best_nn, ax=axes[3])
    axes[3].set_title(f"{target} | Neural Network\\n{best_nn['Features']} | Test F1 = {best_nn['Test F1']:.3f}", fontsize=11, fontweight='bold')

    fig.tight_layout()
    plt.show()
"""
            cell['source'] = [l + '\n' for l in rewritten.split('\n')]
            

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Final models and plotting updated successfully.")
