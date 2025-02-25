"""
Vector Database (VDB) Utilities for Medical Data Retrieval

This module provides functions to process and store medical QA datasets in a FAISS-based 
vector database (VDB). The stored data can be efficiently searched using embeddings.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document

# ================================
# EMBEDDING MODEL INITIALIZATION
# ================================

# Load a lightweight sentence embedding model optimized for retrieval tasks
instructor_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"}  # Utilize GPU if available
)

# ================================
# DATA COLLATION FUNCTIONS
# ================================

def lazy_collate(row, cols):
    """
    Converts a DataFrame row into a structured document format with metadata.

    Args:
        row (pd.Series): A row from the dataset.
        cols (list): Column names of the dataset.

    Returns:
        Document: A structured document object with patient information.
    """
    return Document(
        page_content='\n'.join([f"{cols[i]}: {str(row.values[i])}" for i in range(len(row.values))]),
        metadata={'row': row}
    )

def collate_with_metadata(row):
    """
    Collates input data with metadata, retaining only the primary patient input.

    Args:
        row (pd.Series): A row from the dataset.

    Returns:
        Document: A structured document object containing input data and metadata.
    """
    return Document(
        page_content=f"{row['input']}",
        metadata={"id": row['index']}
    )

def collate_with_metadata_with_output(row, input_name='input', output_name='output'):
    """
    Collates both input and output data into a structured document.

    Args:
        row (pd.Series): A row from the dataset.
        input_name (str): Column name for input data.
        output_name (str): Column name for expected output data.

    Returns:
        Document: A structured document with both input and output information.
    """
    return Document(
        page_content=f"Patient Input: {row[input_name]}\nDiagnosis Output: {row[output_name]}",
        metadata={"id": row['index']}
    )

# ================================
# VECTOR DATABASE CREATION & LOADING
# ================================

def get_any_case_df_as_vdb(df="custom_vdb.vdb", embeddings=instructor_embeddings,
                           collate_func=lazy_collate, db_path="custom_vdb.vdb"):
    """
    Creates or loads a FAISS-based Vector Database (VDB) from a given DataFrame.

    Args:
        df (pd.DataFrame or str): The input DataFrame to be converted into a VDB. If a string is provided, an existing database is loaded.
        embeddings (Embeddings, optional): The embedding model used for vectorization. Defaults to MiniLM.
        collate_func (Callable, optional): The function to format the dataset into a structured document format. Defaults to `lazy_collate`.
        db_path (str, optional): The path where the VDB is saved or loaded from. Defaults to "custom_vdb.vdb".

    Returns:
        FAISS: A FAISS index representing the vectorized database.

    If `db_path` does not exist, a new VDB is created from `df` using the specified embeddings and collation function.
    The VDB is saved locally for future retrieval.
    
    If `db_path` exists, the existing VDB is loaded to enable quick similarity searches.
    """
    if not os.path.exists(db_path):
        if not isinstance(df, str):  # Ensure df is not mistakenly passed as a string
            faiss_index = FAISS.from_documents(df.apply(collate_func, axis=1), embeddings)
            faiss_index.save_local(db_path)  # Save newly created database
    else:
        faiss_index = FAISS.load_local(db_path, embeddings)  # Load pre-existing database

    return faiss_index
