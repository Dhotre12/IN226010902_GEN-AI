from prompts.evaluation_prompt import evaluation_prompt
from utils.config import get_llm
from langchain_core.output_parsers import JsonOutputParser

# Using JsonOutputParser to satisfy the "Structured JSON output" bonus requirement
llm = get_llm(temperature=0.1)
parser = JsonOutputParser()

# LCEL Pipeline
evaluation_chain = evaluation_prompt | llm | parser