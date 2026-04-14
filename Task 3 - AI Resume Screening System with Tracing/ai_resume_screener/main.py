from dotenv import load_dotenv
load_dotenv()
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chains.extraction_chain import extraction_chain
from chains.evaluation_chain import evaluation_chain

def process_resume(resume_text, job_description):
    """
    Executes the Resume Screening Pipeline:
    Resume -> Extract -> Match -> Score -> Explain -> Tracing
    """
    # Step 1: Skill Extraction
    extracted_info = extraction_chain.invoke({"resume": resume_text}).content.strip()

    # Step 2, 3 & 4: Matching Logic, Scoring, and Explanation (Outputs JSON)
    evaluation = evaluation_chain.invoke({
        "job_description": job_description,
        "extracted_info": extracted_info
    })

    return extracted_info, evaluation

if __name__ == "__main__":
    # Job Description
    JOB_DESCRIPTION = """
    Role: Data Scientist
    Requirements: 
    - 2+ years of experience in data science or machine learning.
    - Strong programming skills in Python and SQL.
    - Experience with LLMs and LangChain is highly preferred.
    - Good communication and analytical skills.
    """

    # At least 3 Resumes (Strong, Average, Weak)
    resumes = {
        "Strong Candidate": """Jane Doe. Data Scientist with 3 years of experience. 
        Highly skilled in Python, SQL, and PyTorch. Over the last year, I have built 
        several generative AI pipelines using LangChain and OpenAI APIs. Excellent analytical skills.""",
        
        "Average Candidate": """John Smith. Data Analyst with 1.5 years of experience. 
        Proficient in Python and SQL for data analysis. Created dashboards using Tableau. 
        Familiar with basic machine learning using Scikit-learn, but no experience with GenAI or LangChain yet.""",
        
        "Weak Candidate": """Alice Johnson. Marketing Specialist with 5 years of experience. 
        Expert in content creation, SEO, and social media campaigns. Proficient in MS Office, 
        Excel, and Canva. Excellent communication skills and a team player."""
    }

    # Run the pipeline for each candidate
    for candidate_type, resume_text in resumes.items():
        print(f"========== Evaluating {candidate_type.upper()} ==========")
        
        try:
            extracted_info, evaluation = process_resume(resume_text, JOB_DESCRIPTION)
            
            print("\n--- EXTRACTED INFO ---")
            print(extracted_info)
            
            print("\n--- EVALUATION RESULTS ---")
            print(f"Score: {evaluation.get('score')}/100")
            print(f"Analysis: {evaluation.get('match_analysis')}")
            print(f"Explanation: {evaluation.get('explanation')}\n")
            
        except Exception as e:
            print(f"Error processing {candidate_type}: {str(e)}\n")