from dotenv import load_dotenv
load_dotenv()

import argparse
import os

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langsmith import traceable


def load_documents(path: str):
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding="utf-8")
        return loader.load()

    if os.path.isdir(path):
        loader = DirectoryLoader(
            path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        return loader.load()

    raise ValueError(f"Path not found: {path}")


@traceable(run_type="chain", name="ingest_documents")
def ingest(path: str, db_dir: str, chunk_size: int, chunk_overlap: int):
    print(f"Loading documents from: {path}")
    documents = load_documents(path)
    print(f"Loaded {len(documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Creating vector store...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir,
    )
    print(f"Done. Vector store saved to '{db_dir}'")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument("path", help="Path to file or directory with .txt/.pdf")
    parser.add_argument(
        "--db", default="chroma_db", help="ChromaDB persist directory"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="Chunk size in characters"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=50, help="Chunk overlap in characters"
    )
    args = parser.parse_args()

    ingest(args.path, args.db, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()
