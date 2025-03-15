from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.surprisal_utils import *
import torch

def get_surprisal_tokennorm(prompts_set, model_name):
    """
    Calculate token surprisal scores for a set of prompts using a pre-trained language model.
    Handles both single-token and multi-token words, processing them in per-sample batches.

    Args:
        prompts_set (list): List of dictionaries containing:
            - "truncated snippet": Original text
            - "prompt": Text with [MASK] token(s)
            - "token": Target word to predict
        model_name (str): Name of the reference model to use

    Returns:
        list: List of dictionaries containing:
            - "token prob": Probability of correct token
            - "token rank": Rank of correct token in predictions
            - "candidates": Alternative predictions
            - "candidates_probs": Probabilities of alternatives
            - "p_dist": Full probability distribution
    """

    # Initialize tokenizer and model for masked language modeling
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # Get the ID for the [MASK] token
    mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")

    # Initialize lists to store batch processing data
    data_surprisal = []
    batch_tokens_tensor = []
    batch_segments_tensor = []
    batch_attention_mask = []
    batch_masked_indices = []
    batch_og_token_ids = []
    prompts_final = []
    prompts_to_encode = []

    # Maximum sequence length for input tokenization
    MAX_SEQ_LENGTH = 256

    for each_prompt in prompts_set:
        # Extract text and prompt from input
        text = each_prompt["truncated snippet"]
        prompt = each_prompt["prompt"]

        # Tokenize input text
        tokenized_text = tokenizer.tokenize(text)
        original_tokenized_text = tokenized_text

        # Get the target token and its tokenized form
        masked_token = each_prompt["token"].lower()
        masked_token_form = tokenizer.tokenize(masked_token)

        # Skip if tokenization failed
        if len(masked_token_form) == 0:
            continue

        # Handle single-token words
        if len(masked_token_form) == 1:
            original_token = masked_token
            original_token_id = tokenizer.convert_tokens_to_ids(original_token)
            tokenized_text = tokenizer.tokenize(prompt)

            # Encode prompt with padding to max length
            tokenized_seq_dict = tokenizer.encode_plus(
                prompt, max_length=MAX_SEQ_LENGTH, pad_to_max_length=True)
            
            prompts_to_encode.append(prompt)

            # Extract token IDs, segment IDs, and attention mask
            indexed_tokens = tokenized_seq_dict["input_ids"]
            segments_ids = tokenized_seq_dict["token_type_ids"]
            attention_mask = tokenized_seq_dict["attention_mask"]

            try:
                # Find position of mask token
                masked_index = indexed_tokens.index(mask_token_id)
            except Exception as e:
                print("Error in masked index")
                import pdb
                pdb.set_trace()
                continue


        # Handle multi-token words
        else:
            original_tokens = masked_token_form
            original_token_ids = tokenizer.convert_tokens_to_ids(masked_token_form)
            
            # For words with multiple tokens, we only want to predict the first token
            original_token_id = original_token_ids[0]


            # Replace single [MASK] with multiple [MASK] tokens based on word length
            tokenized_seq_dict = tokenizer.encode_plus(
                prompt.replace("[MASK]", " ".join(["[MASK]"]*len(masked_token_form))), 
                max_length=MAX_SEQ_LENGTH, 
                pad_to_max_length=True
            )
            prompts_to_encode.append(prompt.replace("[MASK]", " ".join(["[MASK]"]*len(masked_token_form))))

            print(tokenized_text)

            # Extract encoded sequence information
            indexed_tokens = tokenized_seq_dict["input_ids"]
            segments_ids = tokenized_seq_dict["token_type_ids"]
            attention_mask = tokenized_seq_dict["attention_mask"]


            # Find start position of masked tokens
            masked_index = get_start_token(indexed_tokens, mask_token_id)
            if not masked_index:
                continue

        # Add to batch processing lists
        batch_tokens_tensor.append(indexed_tokens)
        batch_segments_tensor.append(segments_ids)
        batch_attention_mask.append(attention_mask)
        batch_masked_indices.append(masked_index)
        batch_og_token_ids.append(original_token_id)
        prompts_final.append(each_prompt)

    # Set device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and move to appropriate device
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Process batches if there are any
    if len(batch_tokens_tensor) > 0:
        # Convert batch data to tensors and move to device
        batch_tokens_tensor = torch.tensor(batch_tokens_tensor).to(device)
        batch_segments_tensor = torch.tensor(batch_segments_tensor).to(device)
        batch_attention_mask = torch.tensor(batch_attention_mask).to(device)

        # Run model inference
        with torch.no_grad():
            try:
                outputs = model(
                    batch_tokens_tensor, 
                    token_type_ids=batch_segments_tensor, 
                    attention_mask=batch_attention_mask
                )
            except Exception as e:
                import pdb
                pdb.set_trace()

            predictions = outputs[0]

        # Process predictions for each item in batch
        for ind, masked_index in enumerate(batch_masked_indices):
            each_prompt = prompts_final[ind]

            # Calculate log softmax probabilities for masked position
            logits_softmax = F.log_softmax(predictions[ind, masked_index])
            original_token_id = batch_og_token_ids[ind]

            # Get probability and rank of correct token
            original_token_prob = logits_softmax[original_token_id]
            original_token_rank = get_rank(logits_softmax, original_token_id)

            # Get alternative candidates and their probabilities
            alt_candidates, alt_candidate_probs = get_alt_candidates(
                logits_softmax, original_token_id)

            # Store results in prompt dictionary
            each_prompt["token prob"] = original_token_prob.item()
            each_prompt["token rank"] = original_token_rank.item()
            each_prompt["candidates"] = " | ".join(alt_candidates)
            each_prompt["candidates_probs"] = " | ".join(
                [str(round(x.item(), 2)) for x in alt_candidate_probs])
            each_prompt["p_dist"] = logits_softmax.to("cpu")

            data_surprisal.append(each_prompt)

        return data_surprisal