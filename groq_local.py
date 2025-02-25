"""
Groq-based LLM Initialization

This module initializes a local instance of a ChatGroq model to be used for LLM-based 
question generation and evaluation in medical applications.

- `llm`: Standard deterministic model with temperature = 0.
- `noisy_llm`: A variant with temperature = 0.3, introducing slight randomness for diversity.
"""

from langchain_groq import ChatGroq

# ================================
# STANDARD LLM INSTANCE
# ================================

llm = ChatGroq(
    temperature=0,  # Ensures deterministic responses for consistency in evaluations
    model_name="mixtral-8x7b-32768"  # Specifies the model variant
)

# ================================
# NOISY LLM INSTANCE (FOR VARIABILITY)
# ================================

noisy_llm = ChatGroq(
    temperature=0.3,  # Introduces slight randomness to improve diversity in responses
    model_name="mixtral-8x7b-32768"
)
