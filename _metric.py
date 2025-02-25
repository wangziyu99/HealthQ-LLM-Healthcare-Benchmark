"""
Metric Computation for Privacy and Content Analysis

This module provides functions for:
1. Detecting sensitive patient information in text.
2. Named Entity Recognition (NER)-based medical content search.
3. Sentiment and similarity analysis of text.
4. Computing word-based and semantic similarity scores.
"""

import re
import string
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util

# ================================
# PRIVACY-SENSITIVE TERM DETECTION
# ================================

_cache_dir = "_metrics_cache"

# List of sensitive terms (derived from GPT-4 and medical data privacy standards)
_patient_privacy_terms = [
    "ID", "PatientID", "Medical Record Number", "MRN", "Social Security Number", "SSN",
    "Date of Birth", "DOB", "Address", "Zip Code", "Postal Code", "Phone Number",
    "Email Address", "Insurance Policy Number", "Insurance ID", "Claim Number",
    "National Provider Identifier", "NPI", "Biometric Identifiers", "Full face photos",
    "Emergency Contact", "Billing Information", "Payment History", "Account Number",
    "Appointment Date", "Admission Date", "Discharge Date", "Healthcare Proxy", "DNR"
]

def find_words_in_string(terms_dict, text):
    """
    Searches for sensitive terms within a given text.

    Args:
        terms_dict (list): A list of sensitive terms.
        text (str): The text in which to search.

    Returns:
        dict: A dictionary mapping found terms to their positions in the text.
    """
    found_terms = {}
    for term in terms_dict:
        start_index = 0
        indices = []
        while start_index < len(text):
            idx = text.find(term, start_index)
            if idx != -1:
                indices.append(idx)
                start_index = idx + 1
            else:
                break
        if indices:
            found_terms[term] = indices
    return found_terms

def total_occurrences(terms_list, text):
    """
    Counts occurrences of sensitive terms in a given text.

    Args:
        terms_list (list): A list of sensitive terms.
        text (str): The text to analyze.

    Returns:
        int: The total count of occurrences.
    """
    total_count = 0
    for term in terms_list:
        total_count += text.count(term)
    return total_count

def sensitivity_search(text, terms_list=_patient_privacy_terms):
    """
    Checks for occurrences of privacy-sensitive terms in the text.

    Args:
        text (str): The input text to analyze.
        terms_list (list, optional): List of terms to check. Defaults to `_patient_privacy_terms`.

    Returns:
        int: Count of detected sensitive terms.
    """
    return total_occurrences(terms_list, text)

# ================================
# NAMED ENTITY RECOGNITION (NER) SEARCH
# ================================

# Load pre-trained biomedical NER model for detecting medical entities
_med_ner_tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all", cache_dir=_cache_dir)
_med_ner_model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all", cache_dir=_cache_dir)
_med_ner_pipeline = pipeline('ner', model=_med_ner_model, tokenizer=_med_ner_tokenizer)

def med_ner_search(text, nerpipeline=_med_ner_pipeline):
    """
    Extracts specific medical-related entities (age, sex, clinical events, procedures) from text.

    Args:
        text (str): The medical text input.
        nerpipeline (pipeline, optional): The NER pipeline. Defaults to `_med_ner_pipeline`.

    Returns:
        dict: A dictionary with entity counts for sensitive medical categories.
    """
    result = nerpipeline(text)
    sensitive_types = ["B-Age", "B-Sex", "B-Clinical_event", "B-Therapeutic_procedure"]

    if result:
        return {k: sum(1 for item in result if item.get('entity') == k) for k in sensitive_types}
    return {}

# ================================
# SENTIMENT ANALYSIS
# ================================

def textblob_vector(text):
    """
    Computes sentiment polarity and subjectivity for a given text.

    Args:
        text (str): The input text.

    Returns:
        dict: Polarity (negative/positive) and subjectivity scores.
    """
    textblob_result = TextBlob(text).sentiment
    return {'polarity': textblob_result.polarity, 'subjectivity': textblob_result.subjectivity}

# ================================
# TEXT SIMILARITY ANALYSIS
# ================================

def split_to_words(s):
    """
    Splits a string into words while ignoring punctuation.

    Args:
        s (str): Input text.

    Returns:
        list: List of words extracted from the text.
    """
    return [w for w in re.findall(r"[\w']+|[.,!?;]", s) if w not in string.punctuation]

def per_word_similarity(token_list1, token_list2):
    """
    Computes word-wise similarity between two tokenized lists.

    Args:
        token_list1 (list): First list of words.
        token_list2 (list): Second list of words.

    Returns:
        float: Ratio of matching words (0-1).
    """
    assert len(token_list1) == len(token_list2), "String lengths must match for comparison."
    return sum(1 for i, s1 in enumerate(token_list1) if s1 == token_list2[i]) / len(token_list1)

def word_similarity(s1, s2, tokenizer=split_to_words, length_handler="truncate"):
    """
    Computes word similarity between two text inputs.

    Args:
        s1 (str): First text.
        s2 (str): Second text.
        tokenizer (function, optional): Function to tokenize input. Defaults to `split_to_words`.
        length_handler (str, optional): "truncate" or "pad" to handle different lengths. Defaults to "truncate".

    Returns:
        float: Word similarity score.
    """
    t1 = tokenizer(s1)
    t2 = tokenizer(s2)
    l1, l2 = len(t1), len(t2)

    if length_handler == "truncate":
        l = min(l1, l2)
        t1, t2 = t1[:l], t2[:l]
    elif length_handler == "pad":
        if l1 < l2:
            t1 += [' '] * (l2 - l1)
        else:
            t2 += [' '] * (l1 - l2)
    else:
        raise NotImplementedError("Invalid length handling mode.")

    return per_word_similarity(t1, t2)

# ================================
# SEMANTIC SIMILARITY ANALYSIS
# ================================

# Load a sentence transformer model for semantic similarity
_semantic_model_name = "sentence-transformers/all-mpnet-base-v2"
_semantic_model = SentenceTransformer(_semantic_model_name)

def semantic_similarity(s1, s2, semantic_model=_semantic_model):
    """
    Computes semantic similarity between two text inputs using a sentence embedding model.

    Args:
        s1 (str): First text.
        s2 (str): Second text.
        semantic_model (SentenceTransformer, optional): Embedding model. Defaults to `_semantic_model`.

    Returns:
        float: Cosine similarity score between the two embeddings.
    """
    semantic_embeddings = semantic_model.encode([s1, s2])
    return float(util.dot_score(semantic_embeddings[0], semantic_embeddings[1]))

