from datasets import load_dataset
import string
import csv
import jsonlines
import nltk

def write_csv(data, filename):
    '''
    Writes data to a csv file
    '''

    # Writing to the CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Writing each row
        for row in data:
            writer.writerow(row)


def write_jsonl(data, filename):
    '''
    Writes data to a jsonl file
    '''
    with jsonlines.open(filename, mode='w') as writer:
        for each_data in data:

            writer.write(each_data)

def convert_huggingface_data_to_list_dic(dataset):
    """
    Converts a Hugging Face dataset into a list of dictionaries.
    
    Args:
        dataset: A Hugging Face dataset object
        
    Returns:
        all_data: List of dictionaries, where each dictionary contains the data from one example
    """
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data

def get_bookmia():
    """
    Loads and splits the BookMIA dataset into validation and test sets.
    
    Returns:
        validation_set: List of examples for validation (first 1870 examples)
        test_set: List of examples for testing (remaining examples after 1870)
    """
    dataset = load_dataset("swj0419/BookMIA", split="train")
    bookmia_data = convert_huggingface_data_to_list_dic(dataset)
    validation_set = bookmia_data[:1870]
    test_set = bookmia_data[1870:]
    return validation_set, test_set

def can_be_masked_token(token, pos):
    """
    Determines if a token can be masked for model probing based on specific criteria.
    
    Args:
        token (str): The token to check
        pos (str): The part-of-speech tag for the token
        
    Returns:
        bool: True if the token can be masked, False otherwise
        
    Criteria for masking:
    - Token must be alphanumeric
    - Token cannot be punctuation
    - Token cannot be a single alphabetic character
    - Token cannot be certain special characters (', #, —, ")
    - Token must be a noun (NN*) or cardinal number (CD) based on POS tag
    """

    return token.strip().isalnum() and token.strip() not in string.punctuation and not (token.isalpha() and len(token)==1) and token != '’' and "#" not in token and token != '' and token != '—' and token != '“' and (pos.startswith("NN") or pos.startswith("CD"))

