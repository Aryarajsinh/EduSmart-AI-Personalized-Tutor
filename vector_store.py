"""
EduSmart AI - Vector Store & Syllabus Retrieval Engine
Uses LangChain HuggingFaceEmbeddings + Chroma Vector Store
"""

import os
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv()

import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
SYLLABUS_DIR = os.path.join(os.path.dirname(__file__), "syllabus_data")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_config_key(key: str) -> Optional[str]:
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return None

class SyllabusVectorManager:
    """
    Manages vector ingestion, indexing, and multi-subject retrieval.
    """

    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        self.persist_directory = persist_directory
        print(f"[VectorStore] Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        hf_token = get_config_key("HF_TOKEN") or get_config_key("HUGGINGFACEHUB_API_TOKEN")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu", "token": hf_token},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store: Optional[Chroma] = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Loads existing vector store or builds a new one from syllabus_data."""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print(f"[VectorStore] Loading existing Chroma database from {self.persist_directory}")
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.vector_store = Chroma(
                client=self.client,
                embedding_function=self.embeddings,
                collection_name="edusmart_curriculum"
            )
        else:
            print("[VectorStore] No database found. Building fresh index from syllabus files...")
            self.build_index()

    def build_index(self):
        """Loads all syllabus documents from syllabus_data and embeds into Chroma."""
        documents: List[Document] = []
        syllabus_path = Path(SYLLABUS_DIR)

        if not syllabus_path.exists():
            raise FileNotFoundError(f"Syllabus folder missing at {SYLLABUS_DIR}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=60,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "]
        )

        for file_path in syllabus_path.glob("*.md"):
            subject_name = file_path.stem.lower() # science, mathematics, history
            loader = TextLoader(str(file_path), encoding="utf-8")
            raw_docs = loader.load()

            # Attach metadata for subject filtering & citation
            for doc in raw_docs:
                doc.metadata["subject"] = subject_name
                doc.metadata["source_file"] = file_path.name

            split_docs = text_splitter.split_documents(raw_docs)
            documents.extend(split_docs)
            print(f"[VectorStore] Processed {len(split_docs)} chunks for subject '{subject_name}'")

        if not documents:
            print("[VectorStore] Warning: No documents found to index.")
            return

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            client=self.client,
            collection_name="edusmart_curriculum"
        )
        print(f"[VectorStore] Successfully indexed {len(documents)} total chunks into Chroma.")

    def get_retriever(self, subject: Optional[str] = None, k: int = 3):
        """
        Returns a retriever, optionally filtered by subject for precise curriculum targeting.
        """
        if not self.vector_store:
            self._initialize_vector_store()

        search_kwargs = {"k": k}
        if subject and subject.lower() != "all":
            search_kwargs["filter"] = {"subject": subject.lower()}

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

    def search_curriculum(self, query: str, subject: Optional[str] = None, k: int = 3) -> List[Document]:
        """Convenience method for direct similarity search."""
        retriever = self.get_retriever(subject=subject, k=k)
        return retriever.invoke(query)


if __name__ == "__main__":
    manager = SyllabusVectorManager()
    query = "What is the formula for photosynthesis?"
    results = manager.search_curriculum(query, subject="science")
    print("\n--- Test Search Results ---")
    for i, res in enumerate(results, 1):
        print(f"\n[Result {i}] (Subject: {res.metadata.get('subject')})")
        print(res.page_content)
