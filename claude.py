"""
Claude-3 Model Initialization

This module initializes a ChatAnthropic instance of Claude-3 Opus for use in 
medical question evaluation and generation tasks.
"""

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

# ================================
# CLAUDE-3 MODEL INITIALIZATION
# ================================

claude = ChatAnthropic(
    temperature=0,  # Ensures deterministic responses for consistent evaluation
    model_name="claude-3-opus-20240229"  # Specifies the model variant being used
)
