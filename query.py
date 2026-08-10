from dotenv import load_dotenv
load_dotenv()

import argparse
import os

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langsmith import traceable


def format_docs(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "desconocido")
        parts.append(f"[Fragmento {i} — {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


@traceable(run_type="retriever", name="similarity_search")
def retrieve(query: str, persist_dir: str, k: int, model_name: str):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    return vector_store.similarity_search(query, k=k)


@traceable(run_type="llm", name="generate_answer")
def generate(context: str, question: str) -> str:
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=os.environ["OPENCODE_GO_API_KEY"],
        base_url=os.environ["OPENCODE_GO_BASE_URL"],
        temperature=0.3,
    )
    prompt = f"""Eres un asistente que responde preguntas basandose unicamente en el contexto proporcionado.
Si la respuesta no esta en el contexto, di "No encontre informacion suficiente en los documentos."

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
    response = llm.invoke(prompt)
    return response.content


def main():
    parser = argparse.ArgumentParser(description="Query the RAG vector store")
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--db", default="chroma_db", help="ChromaDB persist directory"
    )
    parser.add_argument(
        "-k", type=int, default=4, help="Number of chunks to retrieve"
    )
    args = parser.parse_args()

    docs = retrieve(
        args.query,
        args.db,
        args.k,
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    if not docs:
        print("No se encontraron fragmentos relevantes.")
        return

    print("=" * 60)
    print("Pregunta:", args.query)
    print("-" * 60)

    context = format_docs(docs)
    answer = generate(context, args.query)

    print("Respuesta:", answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
