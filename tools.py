from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from medialpy_as_tool import medial_search
from df_as_search_db import lazy_collate as collate_func, instructor_embeddings
from functools import partial
from hf_data_no_auth import search_cases_contents

# ================================
# VECTOR DATABASE CASE SEARCH TOOL
# ================================

# Creates a search function that retrieves similar medical cases from the database
merged_case_search = partial(search_cases_contents, db=db_merged, sep='\n', k=10)

# ================================
# TOOL INITIALIZATION
# ================================

def initialize_tools():
    """
    Initializes and returns a set of tools for the conversational health agent.

    Returns:
        list: A list of `Tool` objects including:
            - `similar_case_search_tool`: Searches for similar medical cases in the database.
            - `medial_search_tool`: Retrieves medical abbreviation explanations.
    """
    similar_case_search_tool = Tool(
        name="similar_case_search",
        func=merged_case_search,
        description="Use it to search similar cases for patients in the database",
    )
    
    medial_search_tool = Tool(
        name="medical_abbreviation_search",
        func=medial_search,
        description="Use it to explain medical abbreviations",
    )

    return [similar_case_search_tool, medial_search_tool]

# ================================
# AGENT INITIALIZATION
# ================================

def initialize_agent(tools, llm):
    """
    Initializes a conversational agent with the given tools and an LLM.

    Args:
        tools (list): A list of Tool objects for the agent to use.
        llm (LLMChain): The language model used for reasoning and responses.

    Returns:
        AgentExecutor: A LangChain agent instance that uses Zero-Shot ReAct logic.
    """
    token_limit = 64  # Memory token limit for conversational history
    memory = ConversationBufferMemory(memory_key="chat_history", token_limit=token_limit)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Zero-shot reasoning with tools
        verbose=True,  # Enables detailed logging for debugging
        handle_parsing_errors=True,  # Prevents crashes due to unexpected outputs
        memory=memory,  # Stores chat history for contextual interactions
    )
    
    return agent

"""
Wrapper for Medialpy: A dictionary for medical term abbreviations.
This module provides a function to search for medical abbreviation meanings.
"""

# Import Medialpy library for medical abbreviation lookup
import medialpy

def medial_search(query):
    """
    Searches for the meaning of a medical abbreviation using the Medialpy library.

    Args:
        query (str): The medical abbreviation to be searched.

    Returns:
        str: The meaning of the abbreviation if found, otherwise returns "Not Found".
    """
    if medialpy.exists(query):  # Check if the abbreviation exists in Medialpy
        return medialpy.search(query).meaning  # Retrieve the meaning of the abbreviation
    else:
        return "Not Found"  # Return a default response if the abbreviation is not found

