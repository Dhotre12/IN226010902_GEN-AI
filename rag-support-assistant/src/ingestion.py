from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import *

def ingest_pdf():
    print("📄 Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print(f"✂️  Chunking {len(docs)} pages...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)

    print(f"🔢 Embedding {len(chunks)} chunks...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("💾 Storing in ChromaDB...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )
    print(f"✅ Ingestion complete. {len(chunks)} chunks stored.")
    return vectordb

if __name__ == "__main__":
    ingest_pdf()