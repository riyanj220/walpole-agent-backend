from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import os
from decouple import config

# Read Ollama base URL from environment, fallback to local default if not set
OLLAMA_BASE_URL = config('OLLAMA_BASE_URL')

def load_vectorstore():
    data_dir = Path(__file__).resolve().parent / "data"
    out_path = str(data_dir / "walpole")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    return FAISS.load_local(out_path, embeddings, allow_dangerous_deserialization=True)

# Vectorstore
vectorstore = load_vectorstore()

def get_retriever(filter_type=None, chapter=None, exercise_id=None):
    search_kwargs = {"k": 1, "fetch_k": 3, "filter": {}}
    if filter_type:
        search_kwargs["filter"]["type"] = filter_type
    if chapter:
        search_kwargs["filter"]["chapter"] = chapter
    if exercise_id:
        search_kwargs["filter"]["exercise_id"] = exercise_id
    return vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

# Single LLM (Phi-3)
llm = OllamaLLM(
    model="llama3.2:1b",
    base_url=OLLAMA_BASE_URL,
    streaming=True
)

# Prompt
final_prompt = ChatPromptTemplate.from_template("""
You are a probability tutor helping a student using ONLY the textbook content provided below. 

Rules you MUST follow:
- Always restate the problem or definition as given in the context before solving.
- Solve step by step, explaining reasoning clearly (show formulas, substitutions, and calculations).
- Always cite the exercise_id and/or page number from the context metadata when answering.
- If the question asks about an exercise, ONLY use the chunk with the same exercise_id. 
- If the context does not contain the required information, reply exactly with: 
  "Exercise not found in the book context." 
- Do not guess, hallucinate, or reference material outside the provided context.

Question:
{question}

Context (from textbook):
{context}

Answer (step-by-step with citations):
""")

def run_rag(query, docs):
    # docs is a list of Document
    context = "\n\n".join([d.page_content for d in docs])
    chain = (
        {"context": lambda _: context, "question": RunnablePassthrough()}
        | final_prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query)