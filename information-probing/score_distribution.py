"""
    This script calculates and stores token-level surprisal scores for text datasets.

    The main functionality:
    1. Takes validation and test splits from the BookMIA dataset
    2. For each text sample:
        - Extracts valid prompts 
        - Calculates surprisal scores for each token using BERT
        - Stores the sample and its surprisal scores
    3. Saves the results to pickle files for later analysis

    The surprisal scores indicate how unexpected/surprising each token is according to the model,
    which can be used to analyze model behavior and detect potential memorization.

    Output files:
    - data_surprisal_cnn_validation.pkl: Surprisal scores for validation set
    - data_surprisal_nyt_test.pkl: Surprisal scores for test set
"""

from utils import *
from src.prompt_utils import get_valid_prompts_tokennorm
from src.get_surprisal import get_surprisal_tokennorm
import pickle
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description='Calculate surprisal scores for text datasets')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to output pickle file')
    return parser.parse_args()


def get_probability_distribution(dataset, model_name="bert-base-uncased", out_file="./data_surprisal_cnn_validation.pkl"):
    """
    Calculates and stores surprisal scores for each token in a dataset.

    Args:
        dataset: List of dictionaries containing text samples
        model_name: Name of the model to use for calculating surprisal scores (default: "bert-base-uncased")
        out_file: Path where the output pickle file will be saved (default: "./data_surprisal_cnn_validation.pkl")

    For each sample in the dataset:
    1. Extracts valid prompts using a sliding window
    2. Calculates token-level surprisal scores using the specified BERT model
    3. Stores the original sample and its surprisal scores

    The results are saved as a list of (sample, surprisal_scores) tuples in a pickle file.
    """
    data_surprisal = []

    for each_sample in dataset:
        snippet = each_sample["snippet"]
        valid_prompts = get_valid_prompts_tokennorm(
            snippet, window_size=50)  # Number of words in the context window

        surprisal_scores = get_surprisal_tokennorm(
            valid_prompts, model_name=model_name)

        data_surprisal.append((each_sample, surprisal_scores))

    with open(out_file, 'wb') as f:
        pickle.dump(data_surprisal, f)


if __name__ == "__main__":

    # Replace this with your own dataset, depending on your use case.
    validation_dataset, test_dataset = get_bookmia()
    args = get_args()

    get_probability_distribution(
        test_dataset, model_name="bert-base-uncased", out_file=args.output_file)
