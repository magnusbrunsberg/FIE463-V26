import json

notebook_path = 'solution.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create the two new markdown cells to satisfy the syllabus
pt5_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "**Model Architecture & Scaling Requirements:**\n",
        "\n",
        "*   **Feature Scaling:** Scaling is absolutely required! Ridge, Lasso, and ElasticNet models apply mathematical penalties ($\\mathcal{L}_1$ / $\\mathcal{L}_2$ norms) to regression coefficients. If variables are on entirely different numerical scales (e.g., S&P 500 at 5000 vs. Unemployment Rate at 0.04), the penalty mechanism structurally breaks down. We rigorously apply `StandardScaler()` inside a `Pipeline` to normalize all features strictly using training-fold distributions to prevent data leakage.\n",
        "*   **Hyperparameters:**\n",
        "    *   *Linear Regression:* No hyperparameters required (Basic OLS).\n",
        "    *   *RidgeCV & LassoCV:* Both mathematically depend on the Penalty Strength Hyperparameter (`alpha`). We implement internal cross-validation mechanisms searching a massive multidimensional logarithmic grid of potential alpha candidates dynamically resolving the optimal configuration.\n",
        "    *   *Random Forest & Gradient Boosting:* The primary hyperparameters include learning rate, max depth, and n_estimators. To prioritize convergence and combat overfitting, we manually bound `max_depth` and `max_iter`/`n_estimators` limits.\n"
    ]
}

pt6_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "**Hyperparameters & Scaling Context (Classification):**\n",
        "\n",
        "*   **Feature Scaling:** Scaling is natively required. We configure `StandardScaler` iteratively inside our pipeline structurally because `LogisticRegressionCV` and `MLPClassifier` (Neural Network) rely heavily on standardized distance geometry. Neural network weight gradients will fail to converge if incoming covariate distributions are wildly skewed. While tree-based ensembles (Random Forest, Gradient Boosting) are logically immune to scale distortion, maintaining a universally standardized input pipe harmonizes evaluation without harm.\n",
        "*   **Hyperparameters:**\n",
        "    *   *LogisticRegressionCV:* Cross-validates over the inverse regularization strength parameter array (`C`).\n",
        "    *   *RandomForestClassifier & Gradient Boosting:* Relies on constraining branching (`max_depth`) and total estimators (`n_estimators` or `max_iter`) to prevent memorization of noise in weak-signal tabular data.\n",
        "    *   *MLPClassifier (Neural Network):* Hyperparameters include node density and network depth. We implement dual hidden layers `(64, 32)` balancing sufficient architectural complexity without catastrophic overfitting.\n"
    ]
}

# Find insertion indices
index_pt5 = -1
index_pt6 = -1

for i, cell in enumerate(nb.get('cells', [])):
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if "### 5.1 Environment Setup & Feature Groups" in source:
            index_pt5 = i
        if "### 6.1 Classification Model Setup" in source:
             index_pt6 = i

if index_pt6 != -1:
    nb['cells'].insert(index_pt6, pt6_markdown)

if index_pt5 != -1:
    nb['cells'].insert(index_pt5, pt5_markdown)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Scaling and Hyperparameter markdowns injected successfully.")
