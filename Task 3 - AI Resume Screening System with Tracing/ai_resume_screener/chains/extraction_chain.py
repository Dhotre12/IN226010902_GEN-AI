from prompts.extraction_prompt import extraction_prompt
from utils.config import get_llm

# Temperature 0.0 to ensure strict extraction without creative additions
llm = get_llm(temperature=0.0)

# LCEL Pipeline
extraction_chain = extraction_prompt | llm