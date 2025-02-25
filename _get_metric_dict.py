"""
Metric Computation Utilities

This module provides functions for:
1. Comparing entity sets extracted using Named Entity Recognition (NER).
2. Computing evaluation scores for generated text using ROUGE and semantic similarity.
3. Structuring multiple search and comparison tasks for medical text analysis.
"""

import gc
import pandas as pd
from _metric import *
from rouge import Rouge
from NER_as_tool import extract_medication_and_symptom

# ================================
# ENTITY SET COMPARISON FUNCTIONS
# ================================

def compare_sets(A, B):
    """
    Computes overlap and differences between two sets.

    Args:
        A (set): The first set.
        B (set): The second set.

    Returns:
        dict: A dictionary containing:
            - Intersection count
            - Left and right differences
            - Ratios of shared elements
    """
    A, B = set(A), set(B)

    # Compute intersection and differences
    intersection = A.intersection(B)
    left_difference = A.difference(B)
    right_difference = B.difference(A)

    # Compute counts
    num_common_elements = len(intersection)
    num_left_difference = len(left_difference)
    num_right_difference = len(right_difference)

    # Compute ratios
    inter_left_ratio = num_common_elements / (num_common_elements + num_left_difference) if num_common_elements + num_left_difference > 0 else 0
    inter_right_ratio = num_common_elements / (num_common_elements + num_right_difference) if num_common_elements + num_right_difference > 0 else 0
    inter_all_ratio = num_common_elements / (num_common_elements + num_left_difference + num_right_difference) if num_common_elements + num_left_difference + num_right_difference > 0 else 0

    return {
        "intersection_count": num_common_elements,
        "left_difference_count": num_left_difference,
        "right_difference_count": num_right_difference,
        "inter_left_ratio": inter_left_ratio,
        "inter_right_ratio": inter_right_ratio,
        "inter_all_ratio": inter_all_ratio
    }

def add_total_set(dict_of_sets):
    """
    Adds a 'total' key to a dictionary of sets, containing the union of all sets.

    Args:
        dict_of_sets (dict): Dictionary where values are sets.

    Returns:
        dict: Updated dictionary including a 'total' key with the combined set.
    """
    total_set = set()
    for set_value in dict_of_sets.values():
        total_set |= set_value
    dict_of_sets["total"] = total_set
    return dict_of_sets

def get_NER_set_diff(hypothesis, reference):
    """
    Compares Named Entity Recognition (NER) extracted entities between hypothesis and reference.

    Args:
        hypothesis (str): Generated text.
        reference (str): Ground truth text.

    Returns:
        dict: Comparison results per entity type.
    """
    hypo_set = add_total_set(extract_medication_and_symptom(hypothesis, return_type='set'))
    reference_set = add_total_set(extract_medication_and_symptom(reference, return_type='set'))

    return {
        entity_type: compare_sets(hypo_set[entity_type], reference_set[entity_type])
        for entity_type in hypo_set.keys()
    }

# ================================
# TEXT EVALUATION METRICS
# ================================

# Initialize ROUGE scoring
rouge = Rouge()
rouge_scoring = lambda hypothesis, reference: rouge.get_scores(hypothesis, reference)[0]

# ================================
# TASK DICTIONARIES FOR EVALUATION
# ================================

""" 
Search tasks that take a single string as input 
"""
_search_dict = {
    "sensitivity_search": sensitivity_search,
    "name_search": name_search,
    "med_ner_search": med_ner_search,
    "textblob_vector": textblob_vector
}

""" 
Comparison tasks that take two strings as input
"""
_similarity_dict = {
    "semantic_similarity": semantic_similarity,
    "word_similarity": word_similarity_all
}

""" 
Medical evaluation tasks using ROUGE and NER-based entity comparison
"""
_medagent_dict = {
    "rouge": rouge_scoring,
    "NER2NER": get_NER_set_diff
}

# ================================
# SCORE COMPUTATION FUNCTION
# ================================

def get_score_dict(s1, s2=None, s1_tasks=_search_dict, s2_tasks=_similarity_dict,
                   gc_collect="none", flatten=False):
    """
    Computes multiple scores for medical text comparison.

    Args:
        s1 (str): First text input.
        s2 (str, optional): Second text input for comparison.
        s1_tasks (dict, optional): Dictionary of functions requiring a single input.
        s2_tasks (dict, optional): Dictionary of functions requiring two inputs.
        gc_collect (str, optional): Garbage collection mode ("none", "outer", "inner").
        flatten (bool, optional): If True, flattens the output dictionary for structured analysis.

    Returns:
        dict: A dictionary containing computed scores.
    """
    result = {}

    assert s1 is not None, "s1 (first input string) cannot be None"

    # Compute single-input task results
    if gc_collect == "none":
        for k, v in s1_tasks.items():
            result[k] = v(s1)
        gc.collect()

        # Compute two-input task results if s2 is provided
        if s2 is not None:
            for k, v in s2_tasks.items():
                result[k] = v(s1, s2)

    elif gc_collect == "outer":
        for k, v in s1_tasks.items():
            result[k] = v(s1)
        gc.collect()

        if s2 is not None:
            for k, v in s2_tasks.items():
                result[k] = v(s1, s2)

    elif gc_collect == "inner":
        for k, v in s1_tasks.items():
            gc.collect()
            result[k] = v(s1)

        if s2 is not None:
            gc.collect()
            for k, v in s2_tasks.items():
                result[k] = v(s1, s2)

    # Flatten output for structured representation
    if flatten:
        result = pd.json_normalize(result, sep='_')

    return result
