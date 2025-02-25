"""
Hardcoded Workflow for Medical Questioning System

This module implements a simple rule-based (hardcoded) workflow for guiding a 
patient through symptom and medication verification using a structured decision tree.
"""

import copy
from NER_as_tool import extract_medication_and_symptom

# ================================
# HARDCODED WORKFLOW FUNCTION
# ================================

def hardcoded(bot_input, state, par, local_case_search):
    """
    Implements a hardcoded workflow for extracting and verifying patient symptoms and medications.

    The workflow follows a step-by-step process:
    1. Asks the patient to describe their symptoms.
    2. Retrieves similar cases from a medical database.
    3. Extracts potential symptoms and medications using Named Entity Recognition (NER).
    4. Asks the patient to confirm extracted symptoms one by one.
    5. Asks the patient to confirm extracted medications.
    6. Returns to the initial state after verification.

    Args:
        bot_input (str): The patient's response or symptom description.
        state (dict): The current state of the conversation.
        par (dict): Workflow parameters.
        local_case_search (function): Function for searching similar cases in the database.

    Returns:
        tuple: (Generated output text, Updated conversation state)
    """
    state_new = copy.deepcopy(state)

    # Initial conversation step: Ask the patient to describe symptoms
    if state["step"] == "textual_conversation":
        output = "Please describe your symptom:"
        state_new["step"] = "retrieval"
        state_new["current_subiter"] = 0

    # Retrieval step: Search for related cases and extract relevant symptoms/medications
    elif state["step"] == "retrieval":
        search_result = local_case_search(bot_input + ',' + ','.join(list(state['symptoms']['True'])))

        # Extract symptoms and medications from retrieved cases
        extracted = extract_medication_and_symptom(search_result)
        state['current_retrieved_results'] = extracted
        state_new['current_retrieved_results'] = extracted

        # Generate response based on extracted entities
        output = f"""Related symptoms are: {','.join(extracted['symptoms'])}
        Related medications are: {','.join(extracted['medication'])}"""

        # Move to symptom confirmation step
        state_new["step"] = "confirm_symptoms"

        if len(extracted['symptoms']) > 0:
            output += f""" Do you have the symptom {state['current_retrieved_results']['symptoms'][0]}?
        Answer yes(1), no(2), or unsure(3)"""
        else:
            # If no symptoms are extracted, proceed to medication verification
            state_new['step'] = 'confirm_medication'
            return hardcoded(bot_input, state_new, par)

        state_new['current_subiter'] = 0

    # Confirmation step: Asking the patient about extracted symptoms
    elif state['step'] == "confirm_symptoms":
        num_symptoms = len(state['current_retrieved_results']['symptoms'])

        # If no symptoms are left to confirm, move to medication confirmation
        if num_symptoms == 0:
            state_new['step'] = "confirm_medications"
            return hardcoded(bot_input, state_new, par)

        # Store the patient's confirmation for the symptom
        if state['current_subiter'] > 0:
            state_new['symptoms'][hard_tf_parser(bot_input)].add(state['current_retrieved_results']['symptoms'][state['current_subiter']])

        # If all symptoms have been verified, move to medication confirmation
        if state['current_subiter'] == num_symptoms - 1:
            state_new["step"] = "confirm_medication"
            state_new['current_subiter'] = 0

            if len(state['current_retrieved_results']['medication']) > 0:
                output = f"""Do you have the medication {state['current_retrieved_results']['medication'][0]}?
            Answer yes(1), no(2), or unsure(3)"""
            else:
                # If no medications are found, reset the workflow
                state_new['step'] = 'confirm_medication'
                state_new['current_iter'] += 1
                return hardcoded(bot_input, state_new, par)

        else:
            # Continue symptom verification
            output = f"""Do you have the symptom {state['current_retrieved_results']['symptoms'][state['current_subiter'] + 1]}?
        Answer yes(1), no(2), or unsure(3)"""
            state_new['current_subiter'] += 1

    # Confirmation step: Asking the patient about extracted medications
    elif state['step'] == 'confirm_medication':
        num_medications = len(state['current_retrieved_results']['medication'])

        # If no medications are left to confirm, reset the workflow
        if num_medications == 0:
            state_new['step'] = "textual_conversation"
            state_new['current_iter'] += 1
            return hardcoded(bot_input, state_new, par)

        # Store the patient's confirmation for the medication
        if state['current_subiter'] > 0:
            state_new['medication'][hard_tf_parser(bot_input)].add(state['current_retrieved_results']['medication'][state['current_subiter']])

        # If all medications have been verified, return to initial conversation step
        if state['current_subiter'] == num_medications - 1:
            state_new["step"] = "textual_conversation"
            state_new['current_subiter'] = 0
            state_new['current_iter'] += 1
            output = "Based on these, do you have descriptions for the symptoms?"

        else:
            # Continue medication verification
            output = f"""Do you have the medication {state['current_retrieved_results']['medication'][state['current_subiter'] + 1]}?
        Answer yes(1), no(2), or unsure(3)"""
            state_new['current_subiter'] += 1

    return output, state_new

# ================================
# TEXT PARSING FUNCTION
# ================================

def hard_tf_parser(bot_in: str):
    """
    Parses the patient's response to a symptom/medication verification question.

    Args:
        bot_in (str): User's response to a yes/no/unsure question.

    Returns:
        str: "True" if the user confirms, "False" if they deny, "Unsure" if they are uncertain.
    """
    bot_in = bot_in.lower()
    if 'yes' in bot_in or bot_in == '1':
        return 'True'
    if 'no' in bot_in or bot_in == '2' or bot_in == '0':
        return 'False'
    if 'unsure' in bot_in or 'not sure' in bot_in or bot_in == '3':
        return 'Unsure'
