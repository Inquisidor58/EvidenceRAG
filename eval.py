from dotenv import load_dotenv
load_dotenv()

import os

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langsmith import Client, traceable
from langsmith.evaluation import evaluate

DB_DIR = os.environ.get("CHROMA_DB_DIR", "chroma_db")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TEST_CASES = [
    {
        "question": "¿Cuántas horas duerme un gato?",
        "expected_answer": "15 horas",
        "topic_keywords": ["gato", "duerme", "15 horas"],
    },
    {
        "question": "¿De dónde descienden los perros?",
        "expected_answer": "lobos",
        "topic_keywords": ["perros", "descienden", "lobos", "domesticados", "olfato"],
    },
    {
        "question": "¿Cómo eran vistos los gatos en el antiguo Egipto?",
        "expected_answer": "sagrados",
        "topic_keywords": ["gatos", "Egipto", "sagrados"],
    },
    {
        "question": "¿Qué tan bueno es el olfato de los perros?",
        "expected_answer": "40 veces",
        "topic_keywords": ["perros", "olfato", "40 veces", "potente"],
    },
    {
        "question": "¿Qué animal duerme 15 horas al día?",
        "expected_answer": "gato",
        "topic_keywords": ["gato", "duerme", "15 horas"],
    },
]


def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )


def create_dataset(client: Client, dataset_name: str):
    if client.has_dataset(dataset_name=dataset_name):
        print(f"Dataset '{dataset_name}' ya existe.")
        return

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Evaluacion RAG: precision, recall y calidad de respuesta",
    )

    for tc in TEST_CASES:
        client.create_example(
            inputs={
                "question": tc["question"],
                "topic_keywords": tc["topic_keywords"],
            },
            outputs={"expected_answer": tc["expected_answer"]},
            dataset_id=dataset.id,
        )

    print(f"Dataset '{dataset_name}' creado con {len(TEST_CASES)} ejemplos.")


def target(inputs: dict) -> dict:
    question = inputs["question"]
    topic_keywords = inputs.get("topic_keywords", [])

    vs = get_vector_store()
    docs = vs.similarity_search(question, k=4)

    if not docs:
        return {"output": "No se encontraron fragmentos.", "retrieved_docs": []}

    context = "\n\n".join(
        f"[{doc.metadata.get('source', '?')}]\n{doc.page_content}" for doc in docs
    )

    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=os.environ["OPENCODE_GO_API_KEY"],
        base_url=os.environ["OPENCODE_GO_BASE_URL"],
        temperature=0.0,
    )
    prompt = f"""Responde basandote solo en el contexto. Se breve.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
    response = llm.invoke(prompt)

    retrieved_text = " ".join(doc.page_content.lower() for doc in docs)

    keywords_found = [kw for kw in topic_keywords if kw.lower() in retrieved_text]
    keywords_missing = [kw for kw in topic_keywords if kw.lower() not in retrieved_text]

    relevant_docs = sum(
        1 for doc in docs
        if any(kw.lower() in doc.page_content.lower() for kw in topic_keywords)
    )
    precision = relevant_docs / max(len(docs), 1)
    recall = len(keywords_found) / max(len(topic_keywords), 1)

    return {
        "output": response.content,
        "retrieved_docs": [d.page_content for d in docs],
        "precision": precision,
        "recall": recall,
        "keywords_found": keywords_found,
        "keywords_missing": keywords_missing,
    }


def precision_evaluator(run, example):
    return {
        "key": "precision@docs",
        "score": run.outputs.get("precision", 0),
        "comment": f"Keywords found: {run.outputs.get('keywords_found', [])} | "
                   f"Missing: {run.outputs.get('keywords_missing', [])}",
    }


def recall_evaluator(run, example):
    return {
        "key": "recall@keywords",
        "score": run.outputs.get("recall", 0),
    }


def answer_evaluator(run, example):
    raw = example.outputs
    if isinstance(raw, dict):
        expected = raw.get("expected_answer", "")
    elif isinstance(raw, str):
        expected = raw
    else:
        expected = str(raw)
    expected = expected.lower()
    answer = (run.outputs.get("output") or "").lower()
    score = 1.0 if expected and expected in answer else 0.0
    return {
        "key": "answer_contains_expected",
        "score": score,
        "comment": f"expected='{expected}' in answer={score}",
    }


def main():
    client = Client()
    dataset_name = "rag-precision-recall"

    if client.has_dataset(dataset_name=dataset_name):
        print(f"Dataset '{dataset_name}' ya existe. Eliminando para regenerar...")
        ds = client.read_dataset(dataset_name=dataset_name)
        for example in client.list_examples(dataset_id=ds.id):
            client.delete_example(example.id)
        client.delete_dataset(dataset_id=ds.id)

    create_dataset(client, dataset_name)

    print(f"\nEjecutando evaluacion...\n")

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[precision_evaluator, recall_evaluator, answer_evaluator],
        experiment_prefix="rag-eval",
        max_concurrency=1,
    )

    print("\n" + "=" * 65)
    print("RESULTADOS DE EVALUACION")
    print("=" * 65)

    total_precision = 0
    total_recall = 0
    total_relevance = 0
    n = 0

    for r in results._results:
        run = r["run"]
        question = run.inputs.get("question", "?")
        answer = (run.outputs.get("output") or "")[:100]

        precision = run.outputs.get("precision", 0)
        recall = run.outputs.get("recall", 0)
        found = run.outputs.get("keywords_found", [])
        missing = run.outputs.get("keywords_missing", [])

        answer_score = 0
        for er in r["evaluation_results"]:
            if isinstance(er, dict):
                results_list = er.get("results", [er])
            else:
                results_list = [er] if not isinstance(er, list) else er
            for res in results_list:
                if isinstance(res, str):
                    continue
                key = res.get("key", "?") if isinstance(res, dict) else "?"
                if key == "answer_contains_expected":
                    answer_score = res.get("score", 0) if isinstance(res, dict) else 0

        total_precision += precision
        total_recall += recall
        total_relevance += answer_score
        n += 1

        print(f"\n Q: {question}")
        print(f" A: {answer}...")
        print(f"   precision={precision:.0%}  recall={recall:.0%}  "
              f"relevance={answer_score:.0%}")
        print(f"   keywords: +{found}  -{missing}")

    if n == 0:
        print("\nSin resultados.")
        return

    avg_precision = total_precision / n
    avg_recall = total_recall / n
    avg_relevance = total_relevance / n
    f1 = 2 * avg_precision * avg_recall / max(avg_precision + avg_recall, 0.001)

    print(f"\n{'=' * 65}")
    print("PROMEDIOS")
    print(f"  Precision@docs:     {avg_precision:.1%}")
    print(f"  Recall@keywords:    {avg_recall:.1%}")
    print(f"  Answer relevance:   {avg_relevance:.1%}")
    print(f"  F1 retrieval:       {f1:.1%}")
    print(f"\nPara ver precision/recall en LangSmith:")
    print(f"  1. Abri el link del experimento que aparece arriba")
    print(f"  2. Las columnas 'precision@docs', 'recall@keywords'")
    print(f"     y 'answer_contains_expected' estan en la tabla")
    print(f"  3. Tambien en: smith.langchain.com > Datasets & Testing")


if __name__ == "__main__":
    main()
