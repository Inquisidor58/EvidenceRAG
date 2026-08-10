from dotenv import load_dotenv
load_dotenv()

import os

import chainlit as cl
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langsmith import Client as LsClient
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

DB_DIR = os.environ.get("CHROMA_DB_DIR", "chroma_db")
K_DOCS = int(os.environ.get("RAG_K", "4"))
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """Eres un asistente que responde preguntas basandose unicamente en el contexto proporcionado.
Si la respuesta no esta en el contexto, di "No encontre informacion suficiente en los documentos."

Contexto:
{context}"""


@traceable(run_type="retriever", name="retrieve")
def retrieve_docs(vector_store, query, k=K_DOCS):
    return vector_store.similarity_search(query, k=k)


@traceable(run_type="llm", name="generate")
def generate_answer(context, question):
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=os.environ["OPENCODE_GO_API_KEY"],
        base_url=os.environ["OPENCODE_GO_BASE_URL"],
        temperature=0.3,
    )
    prompt = SYSTEM_PROMPT.format(context=context)
    prompt += "\n\nPregunta: " + question + "\n\nRespuesta:"
    response = llm.invoke(prompt)
    try:
        rt = get_current_run_tree()
        run_id = str(rt.id) if rt else None
    except Exception:
        run_id = None
    return response.content, run_id


@cl.on_chat_start
async def on_chat_start():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        cl.user_session.set("vector_store", vector_store)
        await cl.Message(content="Listo. Preguntame sobre los documentos cargados.").send()
    except Exception as e:
        await cl.Message(content=f"Error al cargar: {e}").send()


@cl.on_message
async def on_message(msg):
    try:
        question = msg.content
        vs = cl.user_session.get("vector_store")

        docs = retrieve_docs(vs, question, k=K_DOCS)

        if not docs:
            await cl.Message(content="No se encontraron fragmentos relevantes.").send()
            return

        context = "\n\n---\n\n".join(
            f"[{doc.metadata.get('source', '?')}]\n{doc.page_content}" for doc in docs
        )

        answer, run_id = generate_answer(context, question)
        cl.user_session.set("last_run_id", run_id)

        answer_msg = cl.Message(content=answer)
        answer_msg.actions = [
            cl.Action(name="rate_up", value="up", label="👍", payload={"run_id": run_id or ""}),
            cl.Action(name="rate_down", value="down", label="👎", payload={"run_id": run_id or ""}),
        ]
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "desconocido").replace("\\", "/")
            answer_msg.elements.append(
                cl.Text(name=f"fuente_{i}", content=f"[{src}]\n{doc.page_content}", display="side")
            )
        await answer_msg.send()

    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()


@cl.action_callback("rate_up")
async def on_rate_up(action):
    try:
        rid = action.payload.get("run_id", "")
        if rid:
            LsClient().create_feedback(run_id=rid, key="user_rating", score=1.0)
    except Exception:
        pass
    await cl.Message(content="Gracias!").send()


@cl.action_callback("rate_down")
async def on_rate_down(action):
    try:
        rid = action.payload.get("run_id", "")
        if rid:
            LsClient().create_feedback(run_id=rid, key="user_rating", score=0.0)
    except Exception:
        pass
    await cl.Message(content="Gracias por tu feedback. Vamos a revisar esta respuesta.").send()
