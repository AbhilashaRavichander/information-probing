from detokenize.detokenizer import detokenize
from datasets import load_dataset
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

def get_start_token(input_ids, masked_token_id):
    return input_ids.index(masked_token_id)

def get_rank(log_dist, original_token_index):
    """
    Calculate the rank of a token in the distribution of logits.

    Args:
        log_dist (torch.Tensor): Logarithmic probability distribution of token predictions.
        original_token_index (int): Index of the original token to find the rank for.

    Returns:
        torch.Tensor: Rank of the original token in the sorted distribution.
    """

    # Returns sorted list
    top_tokens = torch.topk(log_dist, log_dist.shape[0])
    rank = torch.where(top_tokens.indices == original_token_index)[0]

    return rank

def get_alt_candidates(log_dist, original_token_index, model_name="bert-base-uncased"):
    """
    Retrieve alternative token candidates and their probabilities for a given token.

    Args:
        log_dist (torch.Tensor): Logarithmic probability distribution of token predictions.
        original_token_index (int): Index of the original token to find alternatives for.
        model_name (str): Name of the pre-trained model tokenizer to use. Default is "bert-base-uncased".

    Returns:
        tuple:
            - List[str]: Top 10 alternative tokens.
            - torch.Tensor: Corresponding probabilities of the alternative tokens.
    """

    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Returns sorted list
    top_tokens = torch.topk(log_dist, log_dist.shape[0])

    rank = torch.where(top_tokens.indices == original_token_index)[0]
    top_token_probs = top_tokens.values[:rank.item()]
    indices = top_tokens.indices[:rank.item()]
    top_tokens = tokenizer.convert_ids_to_tokens(indices)

    return top_tokens[:10], top_token_probs[:10]

def get_model_alternatives(model, indexed_tokens, segments_ids, attention_mask, masked_index, original_token_id):
    """
    Generate alternative token predictions for a masked token using a model.

    Args:
        model: The pre-trained language model.
        indexed_tokens: List of token indices for the input.
        segments_ids: List of segment IDs.
        attention_mask: Attention mask for the input.
        masked_index: Index of the masked token in the input.
        original_token_id: Token ID of the original masked token.

    Returns:
        List of top 10 alternative tokens.
    """
    outputs=model(torch.tensor([indexed_tokens]), token_type_ids=torch.tensor([segments_ids]), attention_mask=torch.tensor([attention_mask]))
    predictions=outputs[0]
    logits_softmax = F.log_softmax(predictions[0, masked_index])
    original_token_prob = logits_softmax[original_token_id]
    alt_candidates, alt_candidate_probs = get_alt_candidates(logits_softmax, original_token_id)
    return alt_candidates[:10]
