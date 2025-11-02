from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from decouple import config

# =====================================================
# 🔹 Base Configuration
# =====================================================

OLLAMA_BASE_URL = config("OLLAMA_BASE_URL", default="http://localhost:11434")
MODEL_NAME = "llama3.2:1b"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# =====================================================
# 🔹 Vector Store Loader
# =====================================================

def load_vectorstore():
    """Load FAISS vector store from /data/walpole"""
    data_dir = Path(__file__).resolve().parent / "data"
    out_path = str(data_dir / "walpole")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(out_path, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# =====================================================
# 🔹 LLM Setup
# =====================================================

llm = OllamaLLM(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    streaming=True,
)

# =====================================================
# 🔹 RAG Prompt (Unified)
# =====================================================

final_prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable probability tutor. Use only the content provided below.

Guidelines:
- Always quote the relevant part of the context before explaining or solving.
- For exercises or answers, restrict strictly to the same exercise_id.
- For conceptual questions, combine relevant theory + examples.
- If there is not enough context, respond exactly with:
  "Relevant content not found in the book context."

Question:
{question}

Context:
{context}

Step-by-step explanation:
""")

# =====================================================
# 🔹 RAG Execution (Core)
# =====================================================

def run_rag(query: str, docs: list):
    """Run the RAG pipeline: combine context, prompt LLM."""
    if not docs:
        return "Relevant content not found in the book context."

    context = "\n\n".join([d.page_content for d in docs])
    chain = (
        {"context": lambda _: context, "question": RunnablePassthrough()}
        | final_prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query)
