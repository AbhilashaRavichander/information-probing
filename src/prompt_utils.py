"""
Module for generating valid masked token prompts from text snippets,
==================

This module provides functionality for generating masked token prompts from text snippets,
primarily focused on nouns and numbers. 

Dependencies:
    - nltk: Natural Language Toolkit for text processing
    - string: Python's string utilities
    - detokenize library

Main Features:
    - Token validation based on part-of-speech tags
    - Context window-based prompt generation
    - Duplicate token handling in context windows
    - Masking of selected tokens
"""

import nltk
import string
from detokenize.detokenizer import detokenize


def can_be_masked_token(token, pos):
    """
    Determine if a token is eligible for masking based on its characteristics and part of speech.

    Args:
        token (str): The token to evaluate
        pos (str): Part-of-speech tag for the token

    Returns:
        bool: True if the token can be masked, False otherwise

    Notes:
        Tokens are maskable if they:
        - Are not punctuation
        - Don't contain '#'
        - Are not empty strings or specific special characters
        - Are either nouns (NN*) or cardinal numbers (CD)
    """
    return token.strip() not in string.punctuation and token != '’' and "#" not in token and token != '' and token != '—' and token != '“' and (pos.startswith("NN") or pos.startswith("CD"))


def get_attackprompt(snippet, token, window_size):
    """
    Generate a masked prompt by replacing a specific token within a context window.

    Args:
        snippet (str): The full text snippet to process
        token (str): The token to mask
        window_size (int): Size of the context window (in #words)

    Returns:
        tuple[str, str]: A tuple containing:
            - truncated_snippet: The original text within the window
            - prompt: The masked version of the truncated snippet

    Notes:
        The function handles three cases:
        1. Single occurrence of token
        2. Token as a substring with spaces
        3. Fallback to tokenization-based approach
    """

    # Case 1: Find exact token match
    if snippet.count(token) == 1:
        start_token = max(0, snippet.find(token)-(window_size*6))
        end_token = min(snippet.find(token)+(window_size*6), len(snippet))

        truncated_snippet = snippet[start_token:end_token]

        prompt = truncated_snippet.replace(
            '\x00', '').replace(token.strip(), "[MASK]")
    else:

        # Maybe the word is a substring of another word
        # Case 2: Find token with surrounding spaces

        if " "+token+" " in snippet:
            start_token = max(0, snippet.find(" "+token+" ")-(window_size*6))
            end_token = min(snippet.find(" "+token+" ") +
                            (window_size*6), len(snippet))
            truncated_snippet = snippet[start_token:end_token]

            prompt = truncated_snippet.replace(
                '\x00', '').replace(" "+token+" ", " [MASK] ")

        else:  
            # This is so we can avoid detokenizing as much as possible
            # Case 3: Fallback to tokenization-based approach
            words = nltk.word_tokenize(snippet)

            word_index = words.index(token)
            words[word_index] = "[MASK]"

            start_token = max(0, word_index-(window_size))
            end_token = min(word_index+(window_size), len(snippet))

            truncated_snippet = detokenize(words[start_token:end_token])

            prompt = truncated_snippet.replace(
                '\x00', '')

    return truncated_snippet, prompt


def get_all_prompts(snippet, unique_tagged_pos, window_size):
    """
    Generate masked prompts for all valid tokens in the text.

    Args:
        snippet (str): The full text to process
        unique_tagged_pos (list): List of [index, token, pos_tag, decision] lists
        window_size (int): Size of the context window

    Returns:
        list[dict]: List of dictionaries containing:
            - prompt: The masked text
            - truncated snippet: The windowed original text
            - snippet: The full original text
            - token: The masked token
            - decision: Whether the token was valid for masking

    Notes:
        Tracks seen tokens to avoid duplicates and only processes tokens marked as
        valid for masking (decision=True)
    """
    final_prompts = []


    seen_tokens = []

    for each_token_sample in unique_tagged_pos:
        decision = each_token_sample[-1]
        token = each_token_sample[1]

        if decision:

            truncated_snippet, prompt = get_attackprompt(
                snippet, token, window_size)

            if "[MASK]" in prompt and token not in seen_tokens:
                final_prompts.append({"prompt": prompt, "truncated snippet": truncated_snippet,
                                      "snippet": snippet, "token": token, "decision": decision})

            seen_tokens.append(token)
    return final_prompts


def get_unique_prompt_tokens(full_text, tagged_pos, window_size_words=50):
    '''Checks if the token doesnt occur multiple times in a small context window'''
    unique_window = []

    for (ind, word, pos, decision) in tagged_pos:

        start_token = max(0, ind-window_size_words)
        end_token = min(ind+window_size_words, len(tagged_pos))

        words_window = [x[1] for x in tagged_pos[start_token:end_token]]

        if words_window.count(word) > 1:
            decision = False

        unique_window.append([ind, word, pos, decision])

    return unique_window


def get_valid_prompts_tokennorm(prompt, window_size=50):
    """
    Main function to generate masked prompts from text

    Args:
        prompt (str): The input text to process
        window_size (int, optional): Size of context window. Defaults to 50.

    Returns:
        list[dict]: List of dictionaries containing masked prompts and metadata

    Example:
        >>> text = "The quick brown fox jumps over the lazy dog."
        >>> prompts = get_valid_prompts_tokennorm(text)
        >>> print(len(prompts))  # Number of valid masked prompts generated
        >>> print(prompts[0]["prompt"])  # First masked prompt

    Notes:
        1. Tokenizes input text
        2. Applies POS tagging
        3. Identifies valid tokens for masking
        4. Filters for uniqueness in context
        5. Generates final masked prompts
    """

    word_tokens = nltk.word_tokenize(prompt)
    pos = nltk.pos_tag(word_tokens)

    # TUples of the form (ind, token[0], token[1], False)
    tagged_pos = [[ind, x[0], x[1], can_be_masked_token(
        x[0], x[1])] for ind, x in enumerate(pos)]

    # Must be a unique token in the surrounding window
    unique_tagged_pos = get_unique_prompt_tokens(
        prompt, tagged_pos, window_size_words=window_size)

    all_prompts = get_all_prompts(prompt, unique_tagged_pos, window_size)

    return all_prompts
