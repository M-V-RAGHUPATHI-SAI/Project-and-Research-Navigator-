# -*- coding: utf-8 -*-
"""ml_pipeline.py - EXACT original ML pipeline"""

import os
import numpy as np
import pandas as pd
import uuid
from pathlib import Path
from typing import List, Tuple, Any, Dict

# LangChain imports
import langchain
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
    PyMuPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embedding and Vector DB imports
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from sklearn.metrics.pairwise import cosine_similarity

# LLM imports
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()




# ========== EMBEDDING MANAGER CLASS ==========
class EmbeddingManager:
    """EXACT original EmbeddingManager class"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading Embedding model:{self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(
                f"Model loaded successfully. Embedding dimension:{self.model.get_sentence_embedding_dimension()}"
            )
        except Exception as e:
            print(f"Error loading model{self.model_name}:{e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        print(f"Generating the embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generate embeddings with shape: {embeddings.shape}")
        return embeddings

    def get_embeddings_dim(self) -> int:
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.get_sentence_embedding_dimension()


# ========== VECTOR STORE CLASS ==========
class VectorStore:
    """EXACT original VectorStore class"""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "../content/chroma_db",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.collection = None
        self.client = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF documents for rag"},
            )
            print(f"Vector Store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in the Collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("length of Documents should match length of embeddings")

        print(f"Add documents to the vector storage")

        # Prepare data for the chromadb
        ids = []
        metadatas = []
        documents_texts = []
        embeddings_lists = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique id
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Prepare Metadata
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            # Document content
            documents_texts.append(doc.page_content)

            # Embedding
            embeddings_lists.append(embedding.tolist())

        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_texts,
                embeddings=embeddings_lists,
            )
            print(f"Successfully add Documents {len(documents)} to Vector Store")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise


# ========== RAG RETRIEVER CLASS ==========
class RagRetriever:
    """EXACT original RagRetriever class"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retriever(
        self, query: str, top_k: int = 5, score_threshold: float = -0.3
    ) -> List[Dict[str, Any]]:
        print(f"Retrieving documents of query: {query}")
        print(f"Top_k: {top_k}, Score Threshold: {score_threshold}")

        query_embeddings = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embeddings.tolist()],
                n_results=top_k
            )

            retrieved_docs = []

            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    similarity_score = 1 - distance

                    # ✅ SOFT acceptance
                    retrieved_docs.append(
                        {
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1,
                        }
                    )

                print(f"Retrieved {len(retrieved_docs)} Documents (soft match)")
            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"error during retrieval: {e}")
            return []


# ========== DOCUMENT PROCESSING FUNCTIONS ==========
def process_all_docs(doc_directory):
    """Process all PDF and XLSX documents"""
    base_dir = Path(doc_directory)
    all_documents = []

    # ---------- PDF FILES ----------
    pdf_files = list(base_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        print(f"Processing PDF: {pdf_file.name}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
        except Exception as e:
            print(f"PDF error ({pdf_file.name}): {e}")

    # ---------- XLSX FILES ----------
    xlsx_files = list(base_dir.glob("**/*.xlsx"))
    print(f"Found {len(xlsx_files)} XLSX files")

    for xlsx_file in xlsx_files:
        print(f"Processing XLSX: {xlsx_file.name}")
        try:
            df = pd.read_excel(xlsx_file)

            for idx, row in df.iterrows():
                text = " | ".join([f"{col}: {row[col]}" for col in df.columns])

                all_documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source_file": xlsx_file.name,
                            "file_type": "xlsx",
                            "row": idx,
                        },
                    )
                )
        except Exception as e:
            print(f"XLSX error ({xlsx_file.name}): {e}")

    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents



def split_documents(documents, chunk_size=400, chunk_overlap=100):
    """EXACT original split_documents function"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split documents {len(documents)} into {len(split_docs)} chunks")

    if split_docs:
        print(f"Example of a chunk")
        print(f"Content: {split_docs[0].page_content[0:100]} ...")
        print(f"Content: {split_docs[1].page_content[0:100]} ...")
        print(f"Metadata: {split_docs[0].metadata} ...")
    return split_docs


# ========== LLM INITIALIZATION ==========
def initialize_llm():
    """EXACT original LLM initialization"""
    groq_llm_api_key = os.getenv("API_KEY")
    llm = ChatGroq(
        groq_api_key=groq_llm_api_key,
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024,
    )
    return llm


# ========== RAG PIPELINE FUNCTION ==========
def rag_simple(query, retriever, llm, top_k=3):
    """EXACT original rag_simple function"""
    results = retriever.retriever(query, top_k)
    context = "\n\n".join([doc["content"] for doc in results]) if results else ""
    if not context:
        print(f"No relevant context found to answer the question")

    prompt = f"""
You are an intelligent assistant.

Use the context below if it is relevant.
If the context is weak or incomplete, answer using general knowledge,
but prioritize the context when possible.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

    response = llm.invoke([prompt.format(context=context, query=query)])
    return response.content


# ========== MAIN PIPELINE CLASS ==========
class RAGPipeline:
    """Main class to run your EXACT pipeline"""

    def __init__(self):
        self.embedding_manager = None
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.initialized = False

    def initialize(self):
        """Run your EXACT pipeline setup"""
        print("=" * 60)
        print("INITIALIZING RAG PIPELINE (Your Exact Code)")
        print("=" * 60)



        # Step 2: Process all documents
        print("\n2. Processing documents...")
        all_pdf_documents = process_all_docs(
    r"C:\Users\yaswa\Downloads\techSprint\techSprint\techSprint\content"
)

        # Step 3: Split documents
        print("\n3. Splitting documents into chunks...")
        chunks = split_documents(all_pdf_documents)

        # Step 4: Initialize embedding manager
        print("\n4. Initializing embedding manager...")
        self.embedding_manager = EmbeddingManager()

        # Step 5: Initialize vector store
        print("\n5. Initializing vector store...")
        self.vector_store = VectorStore()

        # Step 6: Convert text to embeddings
        print("\n6. Creating embeddings...")
        texts = [doc.page_content for doc in chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)

        # Step 7: Store in vector database
        print("\n7. Storing in vector database...")
        self.vector_store.add_documents(chunks, embeddings)

        # Step 8: Initialize retriever
        print("\n8. Initializing retriever...")
        self.retriever = RagRetriever(self.vector_store, self.embedding_manager)

        # Step 9: Initialize LLM
        print("\n9. Initializing LLM...")
        self.llm = initialize_llm()

        self.initialized = True
        print("\n" + "=" * 60)
        print("✓ RAG PIPELINE INITIALIZED SUCCESSFULLY!")
        print("=" * 60)
        return self

    def query(self, question: str, top_k: int = 3):
        """Run your EXACT query pipeline"""
        if not self.initialized:
            return "Pipeline not initialized. Please run initialize() first.", []

        print(f"\n🔍 QUERY: {question}")

        # Use your EXACT rag_simple function
        answer = rag_simple(question, self.retriever, self.llm, top_k)

        # Also get the retrieved documents for display
        retrieved_docs = self.retriever.retriever(question, top_k)

        return answer, retrieved_docs

    def get_status(self):
        """Get pipeline status"""
        if not self.initialized:
            return {"status": "not_initialized", "documents": 0}

        return {
            "status": "initialized",
            "documents": self.vector_store.collection.count()
            if self.vector_store
            else 0,
        }


# Global instance
_rag_pipeline = None


def get_pipeline():
    """Get or create the pipeline instance"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
        _rag_pipeline.initialize()
    return _rag_pipeline


# ========== TEST IF RUN DIRECTLY ==========
if __name__ == "__main__":
    print("Testing the EXACT ML pipeline...")

    pipeline = get_pipeline()

    # Test with your exact test query
    test_query = "fetch the information regarding previous restaurant management system implemented"
    print(f"\nTesting query: {test_query}")

    answer, docs = pipeline.query(test_query)
    print(f"\nAnswer: {answer}")
    print(f"\nRetrieved {len(docs)} documents")
