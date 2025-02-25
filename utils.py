import re
import time
import copy
import os
import pandas as pd
from functools import partial
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory

# Importing modules for data retrieval, evaluation, and processing
from df_as_search_db import get_any_case_df_as_vdb as get_db
from fake_patient import ground_truth_to_first_answer
from judge import judge
from _get_metric_dict import _medagent_dict, get_score_dict
from NER_as_tool import extract_medication_and_symptom
from hardcoded_workflow import hard_tf_parser, hardcoded
from hf_data_no_auth import search_cases_contents
from medialpy_as_tool import medial_search
from df_as_search_db import lazy_collate as collate_func, instructor_embeddings
from claude import claude

# ================================
# ENVIRONMENT VARIABLE CONFIGURATION
# ================================

# Set API keys for external services (These should be securely stored in production)
os.environ["GROQ_API_KEY"] = "YOUR API KEY"
os.environ["ANTHROPIC_API_KEY"] = "YOUR API KEY"

# ================================
# VECTOR DATABASE INITIALIZATION
# ================================

# Load FAISS-based vector databases for efficient similarity search
db_lavita = get_db(db_path="lavita_train_lazy_vdb")  # Vector database for Lavita dataset
db_mts = get_db(db_path="mts_train_lazy_vdb")  # Vector database for MTS dataset

# Load pre-trained sentence embedding model for document retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})

# Load merged vector database combining multiple datasets
db_merged = FAISS.load_local(folder_path="merged_train_vdb", embeddings=embeddings)

# Define a function for searching cases from the vector database
merged_case_search = partial(search_cases_contents, db=db_merged, sep='\n', k=10)

def search_cases(query, db=db_merged):
    """
    Retrieves the most relevant medical cases for a given query using FAISS-based similarity search.

    Args:
        query (str): The patient statement or symptom description.
        db (FAISS): The vector database for searching relevant cases.

    Returns:
        list: A list of retrieved document contents.
    """
    results = db.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(query)
    return results

def format_retrieval_results(results):
    """
    Formats the retrieved document results into a string for LLM processing.

    Args:
        results (list): List of retrieved documents.

    Returns:
        str: Formatted string of case information.
    """
    return '\n'.join([doc.page_content for doc in results])

# ================================
# FUNCTIONALITY: RETRY MECHANISM
# ================================

def retry_with_timing(*args, n_retry=3, wait_time=1, **kwargs):
    """
    Executes a function with retry logic in case of failures.

    Args:
        *args: Function to be executed followed by its arguments.
        n_retry (int): Number of retry attempts.
        wait_time (int): Time (in seconds) to wait between retries.
        **kwargs: Additional keyword arguments for the function.

    Returns:
        tuple: (Result, Execution time, Number of retries taken)
    """
    retry_count = 0
    start_time = time.time()
    while retry_count < n_retry:
        try:
            result = args[0](*args[1:], **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            return result, execution_time, retry_count
        except Exception as e:
            retry_count += 1
            if retry_count >= n_retry:
                end_time = time.time()
                execution_time = end_time - start_time
                return "failed", execution_time, retry_count
            else:
                time.sleep(wait_time)
    end_time = time.time()
    execution_time = end_time - start_time
    return "failed", execution_time, retry_count

# ================================
# FUNCTIONALITY: TOOL UTILITIES
# ================================

def extract_tool_and_argument(text):
    """
    Extracts tool name and argument from a structured text command.

    Args:
        text (str): Input text containing tool information.

    Returns:
        tuple: (Tool name, Argument)
    """
    lines = text.split('\n')
    tool_name = None
    argument = None
    for line in lines:
        if line.startswith('Tool:'):
            tool_name = re.sub(r'\\+', '', line[5:].strip())
        elif line.startswith('Argument:'):
            argument = re.sub(r'\\+', '', line[9:].strip())
        if tool_name and argument:
            break
    return tool_name, argument

def format_tools(tools):
    """
    Formats the list of available tools as a readable string.

    Args:
        tools (list): List of tool objects.

    Returns:
        str: Formatted tool descriptions.
    """
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

def format_tool_memory(tool_memory):
    """
    Formats previous tool usage history for debugging and analysis.

    Args:
        tool_memory (list): List of tool usage records.

    Returns:
        str: Formatted history of tool invocations.
    """
    return "\n".join([f"Previous Tool: {entry[0]}\nPrevious Argument: {entry[1]}\nOutput: {entry[2]}\n" for entry in tool_memory])

# ================================
# FUNCTIONALITY: EVALUATION & METRICS
# ================================

def evaluate_and_collect_results(test_df, solvers):
    """
    Evaluates different solvers (LLM-based question-generation workflows) and collects results.

    Args:
        test_df (pd.DataFrame): The test dataset containing patient statements and ground truth information.
        solvers (dict): Dictionary mapping solver names to their function implementations.

    Returns:
        pd.DataFrame: A DataFrame containing evaluation metrics and solver outputs.
    """
    times = []
    results = []
    retries = []
    fake_patient_statements = []
    fake_first_answers = []
    sources = []
    apps = []
    interrogate_metrics = []
    summarization_metrics = []
    i_s = []

    for i in range(len(test_df)):
        fake_patient_statement = test_df.iloc[i]['Patient_first_statement']
        patient_known_knowledge = test_df.iloc[i]['Patient_known_knowledge']
        source = test_df.iloc[i]["source"]

        for solver_name, solver_func in solvers.items():
            i_s.append(i)
            apps.append(solver_name)
            print(f"Solver: {solver_name}")
            result, execution_time, retry_count = retry_with_timing(solver_func, fake_patient_statement, n_retry=3, wait_time=120)
            results.append(result)
            times.append(execution_time)
            retries.append(retry_count)
            fake_patient_statements.append(fake_patient_statement)
            sources.append(source)
            interrogate_metrics.append(judge(patient_known_knowledge, fake_patient_statement, result, claude))
            fake_first_answer = ground_truth_to_first_answer(patient_known_knowledge, result, claude)['Patient_Answer']
            fake_first_answers.append(fake_first_answer)
            summarization_metrics.append(get_score_dict(result, patient_known_knowledge, flatten=True, s1_tasks={}, s2_tasks=_medagent_dict))
            print(f"Finished\n")
            print(result, execution_time, retry_count)

    # Combine results into a structured DataFrame
    interrogate_df = pd.DataFrame(interrogate_metrics)
    summary_df = pd.concat(summarization_metrics)
    results_df = pd.concat((interrogate_df.reset_index(), summary_df.reset_index()), axis=1)
    results_df['source'] = sources
    results_df['app'] = apps
    results_df['fake_patient_statement'] = fake_patient_statement
    results_df['output'] = results
    results_df['fake_first_answer'] = fake_first_answers
    results_df['i'] = i_s
    results_df = results_df.drop('index', axis=1)
    return results_df

# ================================
# FUNCTIONALITY: AGENT INITIALIZATION
# ================================

def _initialize_tools():
    return [Tool(name="similar_case_search", func=merged_case_search, description="Search similar cases."),
            Tool(name="medical_abbreviation_search", func=medial_search, description="Explain medical abbreviations.")]

def _initialize_agent(tools, llm):
    return initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True, memory=ConversationBufferMemory(memory_key="chat_history", token_limit=64))
