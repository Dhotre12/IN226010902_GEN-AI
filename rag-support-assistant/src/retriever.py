import warnings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import *

# Mute LangChain's negative relevance score warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vectordb

def retrieve(query: str, k: int = TOP_K):
    db = get_retriever()
    results = db.similarity_search_with_relevance_scores(query, k=k)
    
    # Filter out negative scores to stop the LangChain warning from impacting data
    valid_results = [(doc, score) for doc, score in results if score > 0]
    
    chunks = [{"text": doc.page_content, "metadata": doc.metadata, "score": score}
              for doc, score in valid_results]
    
    # Use the HIGHEST score (best match) to determine confidence, NOT the average
    best_score = chunks[0]["score"] if chunks else 0.0
    
    return chunks, best_score