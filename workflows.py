import copy
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from utils import search_cases, format_retrieval_results, extract_medication_and_symptom, hardcoded

# ===========================
# PROMPT TEMPLATE DEFINITIONS
# ===========================

# Initial question-generation template for the LLM to create an improved question.
initial_template = """
The patient states: {bot_input}.
Based on the following relevant cases:
{formatted_retrieval_results}
Ask a better question to help diagnose any health issues. This may include asking whether the patient relates to symptoms, medications, habits or professions of the searched relevant cases.
The new question should be concise and specific. Please provide only the new question in your response.
"""

# Reflection-based question refinement template.
reflection_template = """
The patient states: {bot_input}.
The initial question generated was: {initial_question}
Based on the following relevant cases:
{formatted_retrieval_results}
Please reflect on the initial question and assess its quality. Consider the following:
- Does the question adequately address the patient's concerns?
- Does the question make proper use of the information from the relevant cases?
- Is there any additional information from the relevant cases that could be incorporated to improve the question?
If the initial question is satisfactory, please output the satisfactory question as is.
If the initial question can be improved, please generate an updated question that better addresses the patient's concerns and incorporates relevant information from the search results.
Reply with the final question only.
"""

# Intermediate reflection for multi-step reasoning.
intermediate_reflection_template = """
The patient states: {bot_input}.
Previous reflections:
{previous_reflections}
Based on the following relevant cases:
{formatted_retrieval_results}
Please reflect on the information provided and consider the following:
- What symptoms or conditions are related to the patient's concerns?
- What additional information from the relevant cases could be useful in diagnosing the patient's health issues?
- Are there any specific questions that could help gather more relevant information from the patient?
Provide your reflections and thoughts without formulating a final question. Focus on identifying relevant information and potential areas to explore further.
Reply with the final question only.
Final question:
"""

# Final reflection step to determine the most refined and diagnostically valuable question.
final_reflection_template = """
The patient states: {bot_input}.
Previous reflections:
{previous_reflections}
Based on the following relevant cases:
{formatted_retrieval_results}
Considering the previous reflections and the relevant cases, please formulate a concise and specific question to help diagnose the patient's health issues. The question should incorporate the most relevant information and aim to gather additional details from the patient to aid in the diagnosis.
Reply with the final question only.
"""

# Initializing Prompt Templates for use in LLM-based question generation.
initial_prompt = PromptTemplate(input_variables=["bot_input", "formatted_retrieval_results"], template=initial_template)
reflection_prompt = PromptTemplate(input_variables=["bot_input", "initial_question", "formatted_retrieval_results"], template=reflection_template)
intermediate_reflection_prompt = PromptTemplate(input_variables=["bot_input", "previous_reflections", "formatted_retrieval_results"], template=intermediate_reflection_template)
final_reflection_prompt = PromptTemplate(input_variables=["bot_input", "previous_reflections", "formatted_retrieval_results"], template=final_reflection_template)


# =========================================
# RETRIEVAL-AUGMENTED GENERATION (RAG) WORKFLOWS
# =========================================

def RAG_default_workflow(bot_input, state, par, qa_chain):
    """
    Default RAG workflow without iterative reflection. Uses a pre-defined retrieval-based question-answering pipeline.
    
    Args:
        bot_input (str): Initial patient statement.
        state (dict): Current conversation state.
        par (dict): Parameters for the workflow.
        qa_chain (RetrievalQA): The retrieval-augmented LLM-based QA chain.
    
    Returns:
        Tuple[str, dict]: Generated question and updated state.
    """
    state_new = copy.deepcopy(state)
    query = f"""
    The patient states: {bot_input}.
    Based on relevant cases, ask for better statement to help diagnose any health issues.
    The new question should be concise and specific. Please provide only the new question in your response.
    """
    result = qa_chain(query)
    output = result['result']
    state_new["current_iter"] += 1
    return output, state_new


def RAG_workflow(bot_input, state, par, llm):
    """
    RAG workflow with a single-pass question generation step.

    Args:
        bot_input (str): Initial patient statement.
        state (dict): Current conversation state.
        par (dict): Parameters for the workflow.
        llm (LLMChain): The LLM model used to generate questions.

    Returns:
        Tuple[str, dict]: Generated question and updated state.
    """
    state_new = copy.deepcopy(state)
    retrieval_results = search_cases(bot_input)
    formatted_retrieval_results = format_retrieval_results(retrieval_results)
    query = initial_prompt.format(bot_input=bot_input, formatted_retrieval_results=formatted_retrieval_results)
    new_question = llm.invoke(query).content
    state_new.update({"new_question": new_question.strip(), "retrieval_results": retrieval_results})
    return new_question.strip(), state_new


def RAG_workflow_reflection(bot_input, state, par, llm):
    """
    RAG workflow with a reflection step to refine the generated question.

    Args:
        bot_input (str): Initial patient statement.
        state (dict): Current conversation state.
        par (dict): Parameters for the workflow.
        llm (LLMChain): The LLM model used for reflection.

    Returns:
        Tuple[str, dict]: Refined question and updated state.
    """
    state_new = copy.deepcopy(state)
    retrieval_results = search_cases(bot_input)
    formatted_retrieval_results = format_retrieval_results(retrieval_results)

    # Generate initial question
    initial_query = initial_prompt.format(bot_input=bot_input, formatted_retrieval_results=formatted_retrieval_results)
    initial_question = llm.invoke(initial_query).content

    # Reflect on and refine the question
    reflection_query = reflection_prompt.format(bot_input=bot_input, initial_question=initial_question, formatted_retrieval_results=formatted_retrieval_results)
    reflection_output = llm.invoke(reflection_query).content

    state_new.update({"question": reflection_output, "retrieval_results": retrieval_results})
    return reflection_output, state_new


def RAG_workflow_reflection_cot(bot_input, state, par, local_llm, num_reflections=3):
    """
    RAG workflow with multiple rounds of reflection using Chain of Thought (CoT).

    Args:
        bot_input (str): Initial patient statement.
        state (dict): Current conversation state.
        par (dict): Parameters for the workflow.
        local_llm (LLMChain): The LLM model used for iterative reflection.
        num_reflections (int): Number of reflection iterations.

    Returns:
        Tuple[str, dict]: Final question and updated state.
    """
    state_new = copy.deepcopy(state)
    retrieval_results = search_cases(bot_input)
    formatted_retrieval_results = format_retrieval_results(retrieval_results)
    initial_query = initial_prompt.format(bot_input=bot_input, formatted_retrieval_results=formatted_retrieval_results)

    current_output = local_llm.invoke(initial_query).content
    previous_reflections = [current_output]

    for i in range(num_reflections - 1):
        reflection_query = (
            intermediate_reflection_prompt.format(bot_input=bot_input, previous_reflections="\n".join(previous_reflections), formatted_retrieval_results=formatted_retrieval_results)
            if i < num_reflections - 2 else 
            final_reflection_prompt.format(bot_input=bot_input, previous_reflections="\n".join(previous_reflections), formatted_retrieval_results=formatted_retrieval_results)
        )
        reflection_output = local_llm.invoke(reflection_query).content
        previous_reflections.append(reflection_output)

    state_new.update({"question": previous_reflections[-1], "retrieval_results": retrieval_results, "reflections": previous_reflections})
    return previous_reflections[-1], state_new
