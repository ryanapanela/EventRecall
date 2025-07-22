# Functions for automated recall scoring
from typing import List
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from skimage import transform
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from segmentation import run_segmentation

def embedding(text: List[str], model_name: str = 'sentence-transformers/LaBSE'):
    """
    Generate embeddings for the events using a specified model.
    Args:
        text (List[str]): List of events to be embedded.
        model_name (str): Name of the model to use for embeddings.

    Returns:
        List[np.ndarray]: List of embeddings for the events.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text)

    return embeddings

def recall_matrix(narrative_events: List[str], recall_events: List[str], model_name: str = 'sentence-transformers/LaBSE') -> pd.DataFrame:
    """
    Calculate the recall matrix between narrative events and recall events.
    Args:
        narrative_events (List[str]): List of narrative events.
        recall_events (List[str]): List of recall events.
        model_name (str): Name of the model to use for embeddings.

    Returns:
        pd.DataFrame: DataFrame containing the recall matrix.
    """
    narrative_embeddings = embedding(narrative_events, model_name)
    recall_embeddings = embedding(recall_events, model_name)

    matrix = np.zeros((len(narrative_embeddings), len(recall_embeddings)))
    for i in range(len(narrative_embeddings)):
        for j in range(len(recall_embeddings)):
            value = spearmanr(narrative_embeddings[i], recall_embeddings[j])
            matrix[i, j] = value.statistic

    return matrix

def recall_score(narrative_events: List[str], recall_events: List[str], model_name: str = 'sentence-transformers/LaBSE'):
    """
    Calculate the recall score between narrative events and recall events.
    Args:
        narrative_events (List[str]): List of narrative events.
        recall_events (List[str]): List of recall events.
        model_name (str): Name of the model to use for embeddings.

    Returns:
        matrix: np.ndarray: The recall matrix.
        matrix_sq: np.ndarray: The square matrix of recall scores.
        recall_score: np.ndarray: The maximum recall score for each narrative event.
        recall_score_index: np.ndarray: The index of the maximum recall score for each narrative event.
        recall_score_diag: np.ndarray: The diagonal recall scores.
        recall_score_rev_diag: np.ndarray: The reverse diagonal recall scores.
    """
    num_narrative_events = len(narrative_events)
    matrix = recall_matrix(narrative_events, recall_events, model_name)

    # resize to square matrix
    matrix_sq = transform.resize(matrix, (num_narrative_events, num_narrative_events))

    # max correlation (event recall score) and corresponding indices across rows
    recall_score = np.max(matrix_sq, axis=1)
    recall_score_index = np.argmax(matrix_sq, axis=1)

    # scores along diagonal and reverse diagonal
    recall_score_diag = np.diagonal(matrix_sq)
    recall_score_rev_diag = np.diagonal(np.flipud(matrix_sq))

    return matrix, matrix_sq, recall_score, recall_score_index, recall_score_diag, recall_score_rev_diag

def evaluate_recall(narrative_path: str = None, recall_path: str = None, narrative_events: list[str] = None, recall_events: list[str] = None, 
                    model_name: str = 'sentence-transformers/LaBSE', segmentation_model='gpt-4', api_key: str = None, generate_plots: bool = False, output_path: str = None) -> pd.DataFrame:
    """
    Evaluate the recall between narrative events and recall events.
    Args:
        narrative_path (str, optional): Path to the narrative file.
        recall_path (str, optional): Path to the recall file.
        narrative_events (list[str], optional): List of narrative events.
        recall_events (list[str], optional): List of recall events.
        model_name (str): Name of the model to use for embeddings.
        segmentation_model (str): Name of the segmentation model to use.
        api_key (str, optional): OpenAI API key for segmentation.
        generate_plots (bool): Whether to generate plots.
    Returns:
        pd.DataFrame: DataFrame with recall results.
    """
    # Use events if provided, otherwise use paths
    if narrative_events is None:
        if narrative_path is None:
            raise ValueError("Either narrative_events or narrative_path must be provided.")
        narrative_events = run_segmentation(narrative_path, model=segmentation_model, api_key=api_key)[0]
    if recall_events is None:
        if recall_path is None:
            raise ValueError("Either recall_events or recall_path must be provided.")
        recall_events = run_segmentation(recall_path, model=segmentation_model, api_key=api_key)[0]

    _, matrix_sq, event_recall_score, recall_score_index, recall_score_diag, recall_score_rev_diag = recall_score(narrative_events, recall_events, model_name)

    results = pd.DataFrame({
        'narrative_event': narrative_events,
        'recall_score': event_recall_score,
        'recall_score_index': recall_score_index,
        'recall_score_diag': recall_score_diag,
        'recall_score_rev_diag': recall_score_rev_diag
    })
    if output_path is not None:
        results.to_csv(output_path, index=False)

    if generate_plots:
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_sq, annot=True, fmt=".2f", cmap='plasma')
        plt.title('Recall Matrix')
        plt.xlabel('Recall Events')
        plt.ylabel('Narrative Events')
        plt.show(block=False)

    return results