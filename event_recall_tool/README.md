# event_recall_tool: Automated Event Segmentation and Recall Scoring

This package provides an easy-to-use Python API and command-line interface (CLI) for automated event segmentation and recall scoring using LLMs. 

## Features
- Run segmentation and recall scoring on your own data files
- Supports OpenAI API for LLM-based analysis
- Outputs results in standard formats (CSV, plots)
- Usable via Python or CLI

## Installation

Clone the repository and install locally:

```bash
pip install .
```

## Usage

### Python API
```python
from event_recall_tool import run_segmentation, run_recall_scoring

run_segmentation(segmentation_input_path, api_key, output_dir)
run_recall_scoring(recall_input_path, api_key, output_dir)
```

### Command Line
```bash
python -m event_recall_tool.cli --segmentation data/segmentation.csv --recall data/recall.csv --api_key sk-... --output results/
```

## Example Data
See the `data/` folder for example input files.

## For Developers
- Main code: `event_recall_tool/`
- Research scripts: `code/`
- Example data: `data/`
