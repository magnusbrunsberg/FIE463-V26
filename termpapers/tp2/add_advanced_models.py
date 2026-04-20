import json

notebook_path = 'solution.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# First pass: add imports
for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "from sklearn.ensemble import RandomForestClassifier" in source_str:
            new_source = []
            for line in cell['source']:
                new_source.append(line)
                if "from sklearn.ensemble import RandomForestClassifier" in line:
                    new_source.append("from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor\n")
            cell['source'] = new_source
            break

# Update part 5 models
part5_models_found = False
for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any("LassoCV(alphas=" in line for line in source):
            # We found the dictionary creation in continuous models
            new_source = []
            for line in source:
                if "'LassoCV': LassoCV(" in line:
                    # add comma if missing
                    if line.rstrip().endswith(','):
                        new_source.append(line)
                    else:
                        new_source.append(line.rstrip('\n') + ',\n')
                    new_source.append("        'GradientBoosting': HistGradientBoostingRegressor(max_iter=200)\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source
            part5_models_found = True

# Update part 6 models
for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any("class_models = ['Logistic Regression', 'Random Forest']" in line for line in source):
            new_source = []
            for line in source:
                if "class_models = ['Logistic Regression', 'Random Forest']" in line:
                    new_source.append("    class_models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']\n")
                elif "elif m_name == 'Random Forest':" in line: # Insert before Random Forest or after it
                    new_source.append(line)
                elif "# Hard predictions" in line:
                    # We inject before this
                    inject = """
        elif m_name == 'Gradient Boosting':
            gb_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', HistGradientBoostingClassifier(max_iter=200, random_state=42))
            ])
            gb_pipe.fit(X_train, y_train)
            fitted = gb_pipe
            best_param = "Default"
"""
                    new_source.extend([l + '\n' for l in inject.strip('\n').split('\n')])
                    new_source.append("\n" + line)
                else:
                    new_source.append(line)
            cell['source'] = new_source

# Update part 6 plotting
for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any("plot_fractional_forecast(best_logit, ax=axes[0])" in line for line in source):
            new_source = []
            for line in source:
                if "best_rf = max(" in line:
                    new_source.append(line)
                elif "key=lambda x: x['Test F1']" in line and "best_rf" not in "".join(new_source[-2:]): # heuristic to find end of best_rf extraction
                    pass # will handle appropriately
                
                # Let's do a strict replacement for plotting block
                pass

            
            # Since the block is very specific, let's just rewrite the whole cell source if found
            rewritten = """for target in optimist_vars:
    target_models = [m for m in master_results_pt6 if m['Target'] == target]

    best_logit = max([m for m in target_models if m['Classifier'] == 'Logistic Regression'], key=lambda x: x['Test F1'])
    best_rf = max([m for m in target_models if m['Classifier'] == 'Random Forest'], key=lambda x: x['Test F1'])
    best_gb = max([m for m in target_models if m['Classifier'] == 'Gradient Boosting'], key=lambda x: x['Test F1'])

    print(f"\\n{target}")
    print(f"Best Logistic: {best_logit['Features']} | Test F1 = {best_logit['Test F1']:.4f}")
    print(f"Best Random Forest: {best_rf['Features']} | Test F1 = {best_rf['Test F1']:.4f}")
    print(f"Best Gradient Boosting: {best_gb['Features']} | Test F1 = {best_gb['Test F1']:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(24, 4))

    plot_fractional_forecast(best_logit, ax=axes[0])
    axes[0].set_title(f"{target} | Logistic Regression\\n{best_logit['Features']} | Test F1 = {best_logit['Test F1']:.3f}", fontsize=10, fontweight='bold')

    plot_fractional_forecast(best_rf, ax=axes[1])
    axes[1].set_title(f"{target} | Random Forest\\n{best_rf['Features']} | Test F1 = {best_rf['Test F1']:.3f}", fontsize=10, fontweight='bold')
    
    plot_fractional_forecast(best_gb, ax=axes[2])
    axes[2].set_title(f"{target} | Gradient Boosting\\n{best_gb['Features']} | Test F1 = {best_gb['Test F1']:.3f}", fontsize=10, fontweight='bold')

    fig.tight_layout()
    plt.show()
"""
            cell['source'] = [l + '\n' for l in rewritten.split('\n')]
            

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Models and plotting updated successfully.")
