import json
import sys
import os

def py_to_ipynb(py_path, ipynb_path):
    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cells = []
    chunks = content.split('# %%')
    for chunk in chunks:
        if not chunk.strip():
            continue
        
        lines = chunk.lstrip('\n').split('\n')
        
        if lines and lines[0].strip().startswith('[markdown]'):
            # Markdown cell
            source = []
            for line in lines[1:]:
                source.append(line + '\n')
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": source
            })
        else:
            # Code cell
            source = []
            for line in lines:
                source.append(line + '\n')
            if not source:
                continue
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source
            })

    notebook = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Created {ipynb_path}")

if __name__ == '__main__':
    py_path = sys.argv[1]
    ipynb_path = sys.argv[2]
    py_to_ipynb(py_path, ipynb_path)
