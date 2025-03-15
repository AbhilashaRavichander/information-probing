"""
This script extracts and analyzes surprising tokens from text samples based on their probability and rank scores.

Key functionality:
1. Extracts tokens from text that are either:
   - Unusually improbable (low probability threshold)
   - Unusually highly ranked (high rank threshold)
2. Processes these tokens to analyze their "surprisal" value
3. Outputs results in JSONL format for reconstructions

Main components:
- Token extraction and filtering
- Probability and rank-based analysis
"""

import datetime
from utils import *
import logging
import argparse
import pickle


def extract_surprise_tokens(test_set, key, attack_threshold, results_file, low=True, n_attack_cutoff=10, text_key="snippet"):
    """
    Extracts tokens from text samples that are considered "surprising" based on a threshold value.

    Args:
        test_set (list): List of text samples to analyze, where each sample contains tokens and their properties
        key (str): The property to use for threshold comparison (e.g. 'token prob' or 'token rank')
        attack_threshold (float): Threshold value to determine if a token is surprising
        results_file (str): Path to output file where results will be written
        low (bool, optional): If True, tokens below threshold are surprising. If False, tokens above threshold are surprising. Defaults to True.
        n_attack_cutoff (int, optional): Maximum number of surprising tokens to extract per sample. Defaults to 10.
        text_key (str, optional): Key to access the text content in samples. Defaults to "snippet".

    Returns:
        list: List of dictionaries containing extracted surprising tokens and their properties for each sample
    """

    all_results_data = []

    for sample_ind, each_sample in enumerate(test_set):
        print("====================================")
        print("Sample:", sample_ind)

        # Drop the last token because the prompts get very messed up (it always favors punctuation before the token that comes there)
        valid_tokens = each_sample[1][:-1]
        text = each_sample[0][text_key]

        # Only pick tokens with sufficient context and that are rarer than the prescribed threshold
        if low:
            attack_token_indices = [(ind, x[key]) for ind, x in enumerate(
                valid_tokens) if x[key] < attack_threshold and can_be_masked_token(x["token"], nltk.pos_tag([x["token"]])[0][1])]
            attack_token_indices.sort(key=lambda x: x[1])
        else:
            attack_token_indices = [(ind, x[key]) for ind, x in enumerate(
                valid_tokens) if x[key] > attack_threshold and can_be_masked_token(x["token"], nltk.pos_tag([x["token"]])[0][1])]
            attack_token_indices.sort(key=lambda x: x[1], reverse=True)

        attack_token_all = attack_token_indices[:n_attack_cutoff]
        attack_token_indices = [x[0] for x in attack_token_all]
        attack_tokens = [valid_tokens[x]["token"]
                         for x in attack_token_indices]

        attack_prompts=[]
        surprise_tokens = []


        for each_token_ind in attack_token_indices:

            prompt = valid_tokens[each_token_ind]["prompt"]
            surprise_tokens.append(
                valid_tokens[each_token_ind]["token"]+" "+str(valid_tokens[each_token_ind][key]))
            attack_prompts.append(prompt)

        all_results_data.append({"id": str(sample_ind), "text": text, "label": each_sample[0]["label"], "surprise tokens": "\n".join(surprise_tokens), "attack tokens": "\n | \n".join(
            attack_tokens), "attack_prompts": "\n | \n".join(attack_prompts)})

    write_jsonl(all_results_data, results_file)
    return all_results_data


def get_args():
    parser = argparse.ArgumentParser(
        description='Probe data with specified thresholds')
    parser.add_argument('--probability_threshold', type=float, default=-12,
                        help='Threshold for token probability (default: -12)')
    parser.add_argument('--rank_threshold', type=int, default=2000,
                        help='Threshold for token rank (default: 2000)')
    parser.add_argument('--distribution_file', type=str,
                        help='Path to the data file')
    parser.add_argument('--output_filestring', type=str,default="surprise_tokens",
                        help='A string which will form the start of the output file name')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    results = pickle.load(open(args.distribution_file, "rb"))

    results_probe = extract_surprise_tokens(results,  'token prob', args.probability_threshold,
                                            results_file=args.output_filestring+"_prob.jsonl", low=True)

    results_probe = extract_surprise_tokens(results, 'token rank', args.rank_threshold,
                                            results_file=args.output_filestring+"_rank.jsonl", low=False)
