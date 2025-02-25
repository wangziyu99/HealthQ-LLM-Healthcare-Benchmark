"""
Module for loading and processing medical QA datasets from Hugging Face that do not require authentication.

This script is used to build a local vector database (VDB) for extracting medical knowledge from an open dataset.
"""

import pandas as pd
from functools import partial
from datasets import load_dataset
from langchain.schema.document import Document

# Import custom modules for vector database processing
from df_as_search_db import get_any_case_df_as_vdb as get_db
from df_as_search_db import lazy_collate, collate_with_metadata, collate_with_metadata_with_output

# ================================
# DATA LOADING & PROCESSING
# ================================

# Load the Lavita Medical QA dataset from Hugging Face (publicly available, no authentication required)
lavita_dataset = load_dataset("lavita/medical-qa-datasets", "all-processed")

# Convert the dataset into a Pandas DataFrame for easy manipulation
lavita_df = pd.DataFrame(lavita_dataset['train']).reset_index()

# ================================
# VECTOR DATABASE INITIALIZATION
# ================================

# Create vector databases (VDBs) using different collation functions for various retrieval needs

# VDB that includes both patient input and diagnostic output
lavita_w_output = get_db(
    lavita_df,
    collate_func=collate_with_metadata_with_output,
    db_path="lavita_with_output_vdb"
)

# VDB that stores only the patient input and case details
lavita = get_db(
    lavita_df,
    collate_func=collate_with_metadata,
    db_path="lavita_cases_vdb"
)

# Lazy-loaded VDB with full output data (suitable for large datasets)
lavita_w_output_lazy = get_db(
    lavita_df,
    collate_func=partial(lazy_collate, cols=lavita_df.columns),
    db_path="lavita_w_output_lazy_vdb"
)

# Lazy-loaded VDB with case details only
lavita_lazy = get_db(
    lavita_df,
    collate_func=partial(lazy_collate, cols=lavita_df.columns),
    db_path="lavita_lazy_vdb"
)

# ================================
# RETRIEVAL FUNCTIONS
# ================================

def search_cases_contents(query, db, k=6, sep='\n'):
    """
    Searches for relevant medical cases in the vector database using FAISS similarity search.

    Args:
        query (str): The search query (e.g., patient symptoms).
        db (FAISS): The vector database containing medical case knowledge.
        k (int, optional): The number of relevant cases to retrieve. Default is 6.
        sep (str, optional): Separator used to format the retrieved cases. Default is newline.

    Returns:
        str: A formatted string of retrieved case content.
    """
    result = db.similarity_search(query, k=k)
    return sep.join(list(map(lambda doc: doc.page_content, result)))

def search_cases_with_output(query, db, df, k=6, id_name='id',
                             template=lambda row: collate_with_metadata_with_output(row).page_content,
                             sep='\n'):
    """
    Searches for relevant medical cases and retrieves their full structured output.

    Args:
        query (str): The search query (e.g., patient symptoms).
        db (FAISS): The vector database containing medical case knowledge.
        df (pd.DataFrame): The original dataset, used to fetch case details.
        k (int, optional): The number of relevant cases to retrieve. Default is 6.
        id_name (str, optional): The column name that uniquely identifies cases. Default is 'id'.
        template (function, optional): A function to format the retrieved case information.
        sep (str, optional): Separator used to format the retrieved cases. Default is newline.

    Returns:
        str: A formatted string of retrieved case content.
    """
    result = db.similarity_search(query, k=k)
    r = ''
    for index in list(map(lambda doc: doc.metadata[id_name], result)):
        row = df[df[id_name] == index]
        r += sep + template(row)
    return r[len(sep):]

