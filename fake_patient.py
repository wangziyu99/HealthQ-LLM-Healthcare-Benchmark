"""
Simulated Patient Response Generation

This module generates realistic patient responses based on ground truth data.
It simulates:
1. The first statement a patient might say to a doctor.
2. The patient's response to a doctor's question.

These functions help in evaluating conversational healthcare models by mimicking real patient interactions.
"""

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain import LLMChain

# Initialize a callback handler for debugging
handler = ConsoleCallbackHandler()

# ================================
# PATIENT STATEMENT GENERATION
# ================================

def ground_truth_to_first_question(patient_known_knowledge, llm):
    """
    Generates a simulated patient’s first statement based on ground truth information.

    Args:
        patient_known_knowledge (str): The patient's known symptoms and medical background.
        llm (LLMChain): The language model used for generating responses.

    Returns:
        str: A plausible first statement a patient might say to a doctor.
    """

    response_schemas = [
        ResponseSchema(
            name="Patient_first_statement",
            description="What patient knows about the symptoms and background information",
            type="string"
        ),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template = """
    "
    {patient_known_knowledge}
    " 
    Above is all the information the patient knows. Using these data, make up a realistic first statement 
    that the patient might say to a doctor in a clinic.

    Imitation of patient: Based on all the provided knowledge, construct the initial patient statement.
    Note that a patient is unlikely to mention all details at once.
    
    {format_instructions}
    """

    prompt = PromptTemplate(
        input_variables=["patient_known_knowledge"],
        template=template,
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm | output_parser

    return chain.invoke(input={"patient_known_knowledge": patient_known_knowledge}, config={'callbacks': [handler]})


# ================================
# PATIENT RESPONSE GENERATION
# ================================

def ground_truth_to_first_answer(patient_known_knowledge, bot_question, llm):
    """
    Generates a simulated patient’s response to a doctor’s question based on ground truth data.

    Args:
        patient_known_knowledge (str): The patient's known symptoms and medical background.
        bot_question (str): The doctor's question.
        llm (LLMChain): The language model used for generating responses.

    Returns:
        str: The patient's simulated answer, considering the knowledge available.
    """

    response_schemas = [
        ResponseSchema(
            name="Patient_Answer",
            description="What the patient can answer about this question based on the patient known knowledge. "
                        "For aspects not covered in the knowledge, respond with uncertainty.",
            type="string"
        ),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template = """
    The patient knows:
    "
    {patient_known_knowledge}
    " 
    Above is all the information the patient knows. Using these data, answer the doctor's question below.

    The doctor asks:
    {bot_question}

    Imitation of patient: 
    - Construct a response based on what the patient knows.
    - If the question asks about something the patient does not know, respond with uncertainty.
    - Note that a patient is unlikely to provide all information in one response.

    {format_instructions}
    """

    prompt = PromptTemplate(
        input_variables=["patient_known_knowledge", "bot_question"],
        template=template,
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm | output_parser

    return chain.invoke(
        input={"patient_known_knowledge": patient_known_knowledge, "bot_question": bot_question},
        config={'callbacks': [handler]}
    )
