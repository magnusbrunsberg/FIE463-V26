import json

notebook_path = 'solution.ipynb'
try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    with open('markdown_extract.txt', 'w', encoding='utf-8') as f:
        for i, cell in enumerate(nb.get('cells', [])):
            if cell['cell_type'] == 'markdown':
                source = "".join(cell['source'])
                if "Part 5" in source or "Part 6" in source or "Part 7" in source or "## 5." in source or "## 6." in source or "## 7." in source:
                    f.write(f"--- CELL {i} ---\n{source}\n\n")

    print("Markdown extracted to markdown_extract.txt")
except Exception as e:
    print(f"Error: {e}")
