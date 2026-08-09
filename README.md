# Transformer from scratch notebook

This repository contains a small notebook-based study of Transformer attention.
The actual code notebooks are kept in a dedicated folder so the project root stays clean.

## Structure

```text
Transformer/
├── README.md
├── .gitignore
└── notebooks/
	├── Self Attention.ipynb
	└── Multi-head Attention.ipynb
```

## What each file is for

- `notebooks/Self Attention.ipynb` demonstrates the attention mechanism step by step.
- `notebooks/Multi-head Attention.ipynb` expands the idea to multi-head attention and tensor reshaping.
- `README.md` explains the repository layout and how the notebooks are organized.
- `.gitignore` keeps notebook cache files and other generated clutter out of version control.

## How to use

1. Open the `Transformer` folder in VS Code.
2. Open one of the notebooks inside `notebooks/`.
3. Run the cells from top to bottom to follow the derivation.

## Notes

- `Self Attention.ipynb` is the best starting point if you want the basic attention flow first.
- `Multi-head Attention.ipynb` builds on the same concepts and assumes you understand the single-head version.
- Notebook output and execution state are not required in git; only the source notebook files should be tracked.
