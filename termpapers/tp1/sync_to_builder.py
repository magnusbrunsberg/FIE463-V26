import json

with open(r'c:\Users\bruma\OneDrive\Skrivebord\FIE463-V26\termpapers\tp1\termpaper1.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

builder_code = """import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_markdown(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\\n" for line in source.split('\\n')]
    })

def add_code(source):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\\n" for line in source.split('\\n')]
    })

"""

for cell in nb['cells']:
    source = "".join(cell['source'])
    if source.endswith('\\n'):
        source = source[:-1]
        
    source = source.replace('\\\\', '\\\\\\\\') 
    source = source.replace('\"\"\"', '\\\"\\\"\\\"')

    if cell['cell_type'] == 'markdown':
        builder_code += f"add_markdown(\"\"\"{source}\"\"\")\n\n"
    elif cell['cell_type'] == 'code':
        builder_code += f"add_code(\"\"\"{source}\"\"\")\n\n"

builder_code += """
# Dump notebook
with open(r'c:\\\\Users\\\\bruma\\\\OneDrive\\\\Skrivebord\\\\FIE463-V26\\\\termpapers\\\\tp1\\\\termpaper1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
"""

with open(r'c:\Users\bruma\OneDrive\Skrivebord\FIE463-V26\termpapers\tp1\tp1_builder.py', 'w', encoding='utf-8') as f:
    f.write(builder_code)

print("Successfully written tp1_builder.py")

# Gen tp1_generator.py from just code blocks
gen_code = ""
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        gen_code += source
        if not source.endswith('\n'):
            gen_code += "\n"
        gen_code += "\n"

with open(r'c:\Users\bruma\OneDrive\Skrivebord\FIE463-V26\termpapers\tp1\tp1_generator.py', 'w', encoding='utf-8') as f:
    f.write(gen_code)

print("Successfully written tp1_generator.py")
