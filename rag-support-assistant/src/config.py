import os
from dotenv import load_dotenv
load_dotenv()

PDF_PATH = "data/knowledge_base.pdf"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "support_kb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
CONFIDENCE_THRESHOLD = 0.10
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")