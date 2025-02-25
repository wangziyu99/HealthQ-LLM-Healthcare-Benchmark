"""
Biomedical Named Entity Recognition (NER) Tool

This module uses a pre-trained NER model (`d4data/biomedical-ner-all`) to extract:
1. Symptoms and diseases from medical text.
2. Medications and procedures from medical text.
"""

from transformers import pipeline

# ================================
# NER MODEL INITIALIZATION
# ================================

# Load the pre-trained biomedical NER model
pipe = pipeline("token-classification", model="d4data/biomedical-ner-all")

# ================================
# TEXT PROCESSING FUNCTIONS
# ================================

def extract_words(structure, text):
    """
    Extracts full words from tokenized text while ensuring proper boundary selection.

    Args:
        structure (list): List of detected NER entities.
        text (str): The original text.

    Returns:
        list: Extracted words from the text.
    """
    words = []

    for entry in structure:
        start = entry['start']
        end = entry['end']

        # Extend start index leftward to capture full word
        while start > 0 and text[start - 1] not in ' .,;!?\n' and text[start - 1] != '-':
            start -= 1

        # Extend end index rightward to capture full word
        while end < len(text) and text[end] not in ' .,;!?\n' and text[end] != '-':
            end += 1

        # Extract and store the full word
        word = text[start:end]
        words.append(word)
        entry['full_word'] = word

    return words

# ================================
# SYMPTOM & MEDICATION EXTRACTION
# ================================

def extract_symptom(data):
    """
    Extracts symptoms and diseases from NER-processed text.

    Args:
        data (list): Output of the NER model.

    Returns:
        list: Extracted symptoms and diseases.
    """
    symptoms = []
    for entry in data:
        entity_name = entry.get('entity', '').lower()
        if 'disease' in entity_name or 'symptom' in entity_name:
            word = entry.get('full_word', '')
            if word:
                symptoms.append(word)
    return symptoms

def extract_medication(data):
    """
    Extracts medications and procedures from NER-processed text.

    Args:
        data (list): Output of the NER model.

    Returns:
        list: Extracted medications and procedures.
    """
    medications = []
    for entry in data:
        entity = entry.get('entity', '').lower()
        if 'medication' in entity or 'procedure' in entity:
            word = entry.get('full_word', '')
            if word:
                medications.append(word)
    return medications

def extract_entities(text, sep=';'):
    """
    Extracts all named entities from the text using the NER model.

    Args:
        text (str): Input text.
        sep (str, optional): Separator for entity output. Defaults to ';'.

    Returns:
        str: Extracted entities as a formatted string.
    """
    return sep.join(extract_words(pipe(text), text))

# ================================
# FORMATTED EXTRACTION FUNCTIONS
# ================================

def extract_symptom_comma_sep(text, pipe=pipe):
    """
    Extracts symptoms and returns them as a comma-separated string.

    Args:
        text (str): Input text.
        pipe (Pipeline, optional): NER model pipeline. Defaults to `pipe`.

    Returns:
        str: Extracted symptoms formatted as a string.
    """
    structure = pipe(text)
    extract_words(structure, text)
    return ','.join(list(set(extract_symptom(structure))))

def extract_medication_comma_sep(text, pipe=pipe):
    """
    Extracts medications and returns them as a comma-separated string.

    Args:
        text (str): Input text.
        pipe (Pipeline, optional): NER model pipeline. Defaults to `pipe`.

    Returns:
        str: Extracted medications formatted as a string.
    """
    structure = pipe(text)
    extract_words(structure, text)
    return ','.join(list(set(extract_medication(structure))))

def extract_medication_and_symptom(text, pipe=pipe, return_type='list'):
    """
    Extracts both symptoms and medications from text.

    Args:
        text (str): Input text.
        pipe (Pipeline, optional): NER model pipeline. Defaults to `pipe`.
        return_type (str, optional): Output format ('list' or 'set'). Defaults to 'list'.

    Returns:
        dict: Extracted symptoms and medications in the specified format.
    """
    structure = pipe(text)
    extract_words(structure, text)
    
    if return_type == 'list':
        return {
            'symptoms': list(set(extract_symptom(structure))),
            'medication': list(set(extract_medication(structure)))
        }
    else:
        return {
            'symptoms': set(extract_symptom(structure)),
            'medication': set(extract_medication(structure))
        }

def extract_medication_and_symptom_as_text(text, pipe=pipe):
    """
    Extracts symptoms and medications and returns them as a formatted text string.

    Args:
        text (str): Input text.
        pipe (Pipeline, optional): NER model pipeline. Defaults to `pipe`.

    Returns:
        str: Extracted symptoms and medications as formatted text.
    """
    structure = pipe(text)
    extract_words(structure, text)

    return "Symptom: " + ','.join(list(set(extract_symptom(structure)))) + '\n' + \
           "Medication: " + ','.join(list(set(extract_medication(structure))))
