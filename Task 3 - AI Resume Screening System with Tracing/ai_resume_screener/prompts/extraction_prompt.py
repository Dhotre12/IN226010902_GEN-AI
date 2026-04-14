from langchain_core.prompts import PromptTemplate

# Step 1 Prompt: Extracts core data from the unstructured resume
extraction_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
You are an expert technical recruiter. Your task is to extract specific information from the provided resume.

Rules:
- Do NOT hallucinate or assume skills that are not explicitly mentioned.
- Keep the output clean and strictly structured.

Resume:
{resume}

Extract the following details:
1. Skills: [List of technical and soft skills]
2. Experience: [Total years of experience and key roles]
3. Tools: [Software, frameworks, or tools mentioned]
"""
)