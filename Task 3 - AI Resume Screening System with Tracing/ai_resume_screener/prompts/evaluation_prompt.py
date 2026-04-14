from langchain_core.prompts import PromptTemplate

# Step 2 Prompt: Evaluates the extracted info against the Job Description
evaluation_prompt = PromptTemplate(
    input_variables=["job_description", "extracted_info"],
    template="""
You are an AI Resume Screening Evaluation System. 
Your task is to compare the candidate's extracted profile against the job description and output a strict JSON evaluation.

Job Description:
{job_description}

Candidate Extracted Info:
{extracted_info}

Rules:
- Do NOT hallucinate. Base your score strictly on the overlap between the extracted info and the JD.
- Output MUST be in valid JSON format.

Output JSON Structure:
{{
    "match_analysis": "Brief 1-2 sentence explanation comparing candidate to requirements.",
    "score": <Integer between 0 and 100 representing the fit>,
    "explanation": "Detailed reasoning for why this score was assigned based on specific missing or matching skills."
}}
"""
)