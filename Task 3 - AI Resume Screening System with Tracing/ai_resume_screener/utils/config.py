import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def get_llm(temperature=0.0):
    # Using low temperature for analytical tasks to prevent hallucinations
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature
    )