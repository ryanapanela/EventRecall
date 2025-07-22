
# AI Segmentation Recall Tool

Automated event segmentation and recall scoring using LLMs and sentence embeddings. Provides both Python API and CLI for flexible usage.

## Features
- Segment events from narrative and recall files using LLMs or transformer models
- Score recall accuracy using embedding-based similarity
- Supports OpenAI API for LLM-based segmentation
- Outputs results as CSV and visual plots
- Usable via Python or CLI

## Installation

Clone the repository and install locally:

```bash
pip install git+https://github.com/ryanapanela/EventRecall.git
```

## Python API Usage

### Main Functions

#### run_segmentation(path: str, model: str = 'gpt-4', api_key: str = None) -> list[str]
Segments events from a text file using the specified model (OpenAI or transformer).

#### evaluate_recall(
    narrative_path: str = None,
    recall_path: str = None,
    narrative_events: list[str] = None,
    recall_events: list[str] = None,
    model_name: str = 'sentence-transformers/LaBSE',
    segmentation_model: str = 'gpt-4',
    api_key: str = None,
    generate_plots: bool = False,
    output_path: str = None
) -> pd.DataFrame
Scores recall accuracy between narrative and recall events. Accepts file paths or lists of events. Generates a CSV and/or heatmap plot if requested.

#### recall_score(narrative_events: list[str], recall_events: list[str], model_name: str = 'sentence-transformers/LaBSE')
Returns recall matrix, square matrix, max scores, indices, diagonal and reverse diagonal scores.

#### recall_matrix(narrative_events: list[str], recall_events: list[str], model_name: str = 'sentence-transformers/LaBSE')
Returns the full similarity matrix between narrative and recall events using embeddings.

#### embedding(text: list[str], model_name: str = 'sentence-transformers/LaBSE')
Returns sentence embeddings for a list of events.

### Example
```python
from segmentation import run_segmentation
from recall import evaluate_recall

# Segment events
narrative_events = run_segmentation('test/Run.txt', model='gpt-4o-mini', api_key='sk-...')
recall_events = run_segmentation('test/Recall.txt', model='gpt-4o-mini', api_key='sk-...')

# Score recall
# If narratives and recall are already segmented...
results = evaluate_recall(
    narrative_events=narrative_events,
    recall_events=recall_events,
    model_name='sentence-transformers/LaBSE',
    generate_plots=True,
    output_path='recall_results.csv'
)
print(results)

# If narratives are already segmented, but recall is not...
results = evaluate_recall(
    narrative_events=narrative_events,
    recall_path='recall.txt',
    model_name='sentence-transformers/LaBSE',
    generate_plots=True,
    output_path='recall_results.csv'
)
print(results)

```

## Output
- CSV file with recall scores for each event
- Optional heatmap plot of recall matrix

## Example Data
See the `test/` folder for example input files.

## File Structure
- Main code: `segmentation.py`, `recall.py`, `utils.py`
- Example/test data: `test/`
- Setup: `setup.py`

## Function Reference

### segmentation.segmentation(path, model, api_key)
- Segments events from a file using LLM or transformer model
- Returns: DataFrame with events, event boundaries, and list of event strings

### recall.evaluate_recall(...)
- Scores recall between narrative and recall events
- Accepts file paths or lists of events
- Returns: DataFrame with scores
- Optional: generates plot and/or CSV

### recall.recall_score(...)
- Returns recall matrix, square matrix, max scores, indices, diagonal and reverse diagonal scores

### recall.recall_matrix(...)
- Returns similarity matrix between events

### recall.embedding(...)
- Returns sentence embeddings for events

## Developer Info
- Main code: `segmentation.py`, `recall.py`, `utils.py`
- Example/test data: `test/`
- Setup: `setup.py`

## Requirements
- Python 3.8+
- pandas, numpy, scipy, scikit-image, matplotlib, seaborn, sentence-transformers
- For LLM segmentation: OpenAI API key


## License
MIT License

---
**Note:** This release supports only the Python API. CLI functionality may be added in a future version.
