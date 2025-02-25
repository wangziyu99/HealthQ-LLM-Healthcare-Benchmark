"""
Evaluation Script for Ground Truth Data Processing (Lavita Dataset)

This script performs:
1. Data preparation for the Lavita dataset.
2. Vector database (VDB) initialization for retrieval-based workflows.
3. Ground truth data extraction and structuring using an LLM (Claude-3).
4. A retry mechanism to ensure robust data processing.

The goal is to construct well-structured ground truth information for evaluating LLM-based medical applications.
"""

import os
import pandas as pd
from functools import partial
from datasets import load_dataset

# Importing required modules for data processing
from claude import claude  # Claude-3 LLM for data parsing
from groq_local import llm as local_llm  # Local LLM for inference

# ================================
# ENVIRONMENT VARIABLES
# ================================

# API keys for external LLM services (should be securely managed in production)
os.environ["GROQ_API_KEY"] = "YOUR API KEY"
os.environ["ANTHROPIC_API_KEY"] = "YOUR API KEY"

# ================================
# DATA LOADING & SPLITTING
# ================================

first_run = False  # Flag to control dataset reloading

if first_run:
    # Load dataset from Hugging Face
    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
    df = pd.DataFrame(dataset['train'])
    df.to_csv("lavita_chatdoctor_note.csv", index=False)

    # Perform train-test split with a fixed random seed for reproducibility
    import numpy as np
    from sklearn.model_selection import train_test_split

    np.random.seed(3)
    train_indices, test_indices = train_test_split(df.index, test_size=64, random_state=3)

    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]

    # Save train and test splits
    train_df.to_csv("train_lavita_chatdoctor_note.csv", index=False)
    test_df.to_csv("test_lavita_chatdoctor_note.csv", index=False)

else:
    # Load pre-split training and testing datasets
    train_df = pd.read_csv("train_lavita_chatdoctor_note.csv")
    test_df = pd.read_csv("test_lavita_chatdoctor_note.csv")

# Reset index for consistency
train_df = train_df.reset_index()

# ================================
# VECTOR DATABASE INITIALIZATION
# ================================

from df_as_search_db import get_any_case_df_as_vdb as get_db
from df_as_search_db import lazy_collate, collate_with_metadata, collate_with_metadata_with_output

# Construct various versions of the vector database (VDB) for retrieval

# VDB storing both input and output (for retrieval-augmented generation)
train_w_output = get_db(
    train_df,
    collate_func=collate_with_metadata_with_output,
    db_path="lavita_train_with_output_vdb"
)

# VDB storing only input cases (for knowledge retrieval without model bias)
train = get_db(
    train_df,
    collate_func=collate_with_metadata,
    db_path="lavita_train_cases_vdb"
)

# Lazy-loading versions of the above for memory efficiency
train_w_output_lazy = get_db(
    train_df,
    collate_func=partial(lazy_collate, cols=train_df.columns),
    db_path="lavita_train_w_output_lazy_vdb"
)

train_lazy = get_db(
    train_df,
    collate_func=partial(lazy_collate, cols=train_df.columns),
    db_path="lavita_train_lazy_vdb"
)

# ================================
# GROUND TRUTH PARSING USING LLM
# ================================

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import StdOutCallbackHandler
from langchain import LLMChain

# Initialize a callback handler for debugging
handler = StdOutCallbackHandler()

def df_row_to_ground_truth_parse_control(question, llm):
    """
    Parses raw patient consultation data into structured ground truth format.

    Args:
        question (str): The raw patient note text.
        llm (LLMChain): The LLM used for parsing the data.

    Returns:
        dict: Processed patient knowledge and doctor question statements.
    """

    response_schemas = [
        ResponseSchema(
            name="Patient_known_knowledge",
            description="Extracted patient knowledge about symptoms and medical background.",
            type="string"
        ),
        ResponseSchema(
            name="Doctor_question_statements",
            description="All medical questions and statements made by the doctor.",
            type="string"
        ),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    # Define the LLM prompt template for structured extraction
    template = """
    "
    {question}
    "
    Above is a patient note. Process this data and extract the following structured information:
    {format_instructions}
    """

    prompt = PromptTemplate(
        input_variables=["question"],
        template=template,
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm | output_parser

    return chain.invoke(input={"question": question}, config={'callbacks': [handler]})

# ================================
# BATCH PROCESSING & ERROR HANDLING
# ================================

def process_function(i):
    """
    Processes a single test case to extract structured ground truth.

    Args:
        i (int): Index of the test sample.

    Returns:
        dict: Extracted patient information and doctor questions.
    """
    return df_row_to_ground_truth_parse_control(
        lazy_collate(test_df.iloc[i, :], cols=test_df.columns), llm=claude
    )

def process_with_retries(indices, n_retries, unit_func, fallback="failed"):
    """
    Executes a function with retry logic to handle potential failures.

    Args:
        indices (list): List of indices to process.
        n_retries (int): Maximum retry attempts per case.
        unit_func (function): Function to process individual cases.
        fallback (str/dict, optional): Default value if all retries fail.

    Returns:
        list: Processed results with original indices.
    """
    results = []
    for index in indices:
        retry_count = 0
        while retry_count <= n_retries:
            try:
                result = unit_func(index)
                results.append([index, result])
                break
            except Exception as e:
                retry_count += 1
                if retry_count > n_retries:
                    print((index, "failed"))
                    results.append([index, fallback])
                    break
    return results

# ================================
# EXECUTION & RESULT STORAGE
# ================================

# Run the structured extraction process on all test cases with retries
n_retries = 3
results = process_with_retries(
    range(len(test_df)), n_retries, process_function,
    fallback={"Patient_known_knowledge": "failed", "Doctor_question_statements": "failed"}
)

# Convert extracted results into a DataFrame and save
ground_truth_test_df_claude = pd.DataFrame([results[i][1] for i in range(len(results))])
ground_truth_test_df_claude.to_csv("lavita_ground_truth.csv", index=False)

# Load saved ground truth data (for verification)
ground_truth_test_df_claude = pd.read_csv("lavita_ground_truth.csv")
