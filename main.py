"""
Main script for evaluating different LLM-based medical question-generation workflows.
This script loads test data, initializes tools and agents, and evaluates multiple solvers.
"""

import pandas as pd
import time
import copy
import os
from workflows import *
from utils import *
from utils import _initialize_tools, _initialize_agent
from groq_local import llm as local_llm

# ================================
# DATA LOADING & CONFIGURATION
# ================================

# Load the test dataset containing patient statements and ground truth responses
test_df = pd.read_csv("data/merged_ground_truth.csv")

# Set API keys for external services (should be securely managed in production)
os.environ["GROQ_API_KEY"] = "YOUR API KEY"
os.environ["ANTHROPIC_API_KEY"] = "YOUR API KEY"

# ================================
# INITIALIZATION
# ================================

# Initialize external tools and agent
tools = _initialize_tools()
agent = _initialize_agent(tools, local_llm)

# Define the initial conversation state structure
state_init = {
    'symptoms': {'True': set(), 'False': set(), 'Unsure': set()},  # Tracks identified symptoms
    'medication': {'True': set(), 'False': set(), 'Unsure': set()},  # Tracks identified medications
    'processed_entities': [],  # Stores extracted medical entities
    'current_iter': 0,  # Tracks the number of question iterations
    'current_subiter': 0,  # Tracks the number of sub-iterations within a retrieval cycle
    'received_answers': [],  # Stores answers received from the virtual patient
    'current_retrieved_results': None,  # Stores the most recent retrieved case results
    'step': "retrieval"  # Defines the current workflow step
}

# ================================
# SOLVERS DEFINITION
# ================================

# Define different solvers representing various LLM-based workflows for question generation.
solvers = {
    "hardcoded": lambda statement: hardcoded(statement, copy.deepcopy(state_init), {'max_iter': 3}, merged_case_search)[0],
    
    "RAG_default_workflow": lambda statement: RAG_default_workflow(statement, copy.deepcopy(state_init), {'max_iter': 3}, qa_chain_default)[0],
    
    "RAG_workflow": lambda statement: RAG_workflow(statement, copy.deepcopy(state_init), {'max_iter': 3})[0],
    
    "RAG_workflow_reflection": lambda statement: RAG_workflow_reflection(statement, copy.deepcopy(state_init), {'max_iter': 3}, local_llm)[0],
    
    "RAG_workflow_reflection_cot": lambda statement: RAG_workflow_reflection_cot(statement, copy.deepcopy(state_init), {'max_iter': 3}, local_llm)[0],
    
    "RAG_workflow_reflection_cot_sc": lambda statement: RAG_workflow_reflection_cot_sc(statement, copy.deepcopy(state_init), {'max_iter': 3}, local_llm, noisy_llm)[0],
    
    "ReAct_workflow": lambda statement: ReAct_workflow(statement, copy.deepcopy(state_init), {'max_iter': 3}, qa_chain_default, local_llm, agent)[0],
}

# ================================
# EVALUATION & RESULTS STORAGE
# ================================

# Evaluate all solvers using the test dataset and collect results
results_df = evaluate_and_collect_results(test_df, solvers)

# Save evaluation results to a CSV file
results_df.to_csv('results.csv', index=False)
