from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from decouple import config

# =====================================================
# 🔹 Base Configuration
# =====================================================
GROQ_API_KEY = config("GROQ_API_KEY", default=None) 
MODEL_NAME = "llama-3.1-8b-instant" 
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# =====================================================
#  Vector Store Loader
# =====================================================

def load_vectorstore():
    """Load FAISS vector store from /data/walpole"""
    data_dir = Path(__file__).resolve().parent / "data"
    out_path = str(data_dir / "walpole")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(out_path, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# =====================================================
# LLM Setup
# =====================================================

if not GROQ_API_KEY:
    print("="*50)
    print("ERROR: GROQ_API_KEY not found in .env file.")
    print("Please add your Groq API key to the .env file.")
    print("="*50)
    llm = None 
else:
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.1,
        max_tokens=1024, 
    )

# =====================================================
#  RAG Prompt (Unified)
# =====================================================

final_prompt = ChatPromptTemplate.from_template("""You are a probability and statistics tutor helping a student understand textbook content.

STRICT GUIDELINES:
1. Base your answer ONLY on the context provided below
2. If the context contains the exercise/answer/example, present it clearly
3. For theory questions, synthesize information from multiple context sections
4. DO NOT add information not present in the context
5. If context is insufficient, say: "The provided context doesn't contain enough information to answer this question."
6. Use clear, educational language

STUDENT QUESTION:
{question}

TEXTBOOK CONTEXT:
{context}

YOUR ANSWER (based strictly on the context above):
""")

# =====================================================
#  RAG Execution (Core)
# =====================================================

def run_rag(query: str, docs: list):
    """Run the RAG pipeline: combine context, prompt LLM."""
    if not docs:
        return "No relevant content found in the textbook for your query."

    # Combine context with source info
    context_parts = []
    for i, d in enumerate(docs[:5], 1): 
        metadata = d.metadata
        source_info = f"[Source {i}: Chapter {metadata.get('chapter', '?')}, Type: {metadata.get('type', 'unknown')}]"
        context_parts.append(f"{source_info}\n{d.page_content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    print(f"[RAG] Running with {len(docs)} documents, context length: {len(context)} chars")
    
    try:
        chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | final_prompt
            | llm
            | StrOutputParser()
        )
        response = chain.invoke(query)
        print(f"[RAG] Generated response length: {len(response)} chars")
        return response
    except Exception as e:
        print(f"[RAG ERROR] {str(e)}")
        return f"Error generating response: {str(e)}"


def run_general_chat(query: str):
    """
    Handle casual conversation, study tips, and emotional support.
    Does NOT use the vector database.
    """
    # A specific prompt for being a friendly companion
    general_prompt = ChatPromptTemplate.from_template("""
    You are 'Stats Advisor', a cheerful, empathetic, and encouraging study companion for a Probability & Statistics student.
    
    YOUR PERSONALITY:
    - You are friendly, warm, and slightly enthusiastic.
    - You understand that statistics can be hard and stressful.
    - You offer good general study advice (like taking breaks, practicing problems).
    
    YOUR TASK:
    - Reply to the student's casual comment or question.
    - If they are stressed, validate their feelings and offer encouragement.
    - If they say hello, welcome them warmly.
    - DO NOT try to explain complex math concepts in this mode (unless they specifically asked).
    - Keep the response concise (2-3 sentences).

    Student: {question}
    
    Your Friendly Response:
    """)

    try:
        chain = general_prompt | llm | StrOutputParser()
        response = chain.invoke({"question": query})
        return response
    except Exception as e:
        print(f"[GENERAL CHAT ERROR] {str(e)}")
        return "I'm here to help you study! Let's take a deep breath and tackle some statistics."

# =====================================================
#  Diagnostic Functions
# =====================================================

def check_exercise_exists(exercise_id: str) -> dict:
    """Check if an exercise exists in the vectorstore."""
    matches = [
        d for d in vectorstore.docstore._dict.values()
        if d.metadata.get("exercise_id") == exercise_id
    ]
    
    return {
        "exercise_id": exercise_id,
        "found": len(matches) > 0,
        "count": len(matches),
        "metadata": [d.metadata for d in matches]
    }


def check_answer_exists(exercise_id: str) -> dict:
    """Check if an answer exists in the vectorstore."""
    matches = [
        d for d in vectorstore.docstore._dict.values()
        if d.metadata.get("answer_id") == exercise_id
    ]
    
    return {
        "exercise_id": exercise_id,
        "found": len(matches) > 0,
        "count": len(matches),
        "metadata": [d.metadata for d in matches]
    }


def get_stats() -> dict:
    """Get vectorstore statistics."""
    all_docs = list(vectorstore.docstore._dict.values())
    
    type_counts = {}
    chapter_counts = {}
    
    for doc in all_docs:
        doc_type = doc.metadata.get('type', 'unknown')
        chapter = doc.metadata.get('chapter', 'unknown')
        
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        chapter_counts[chapter] = chapter_counts.get(chapter, 0) + 1
    
    return {
        "total_documents": len(all_docs),
        "by_type": type_counts,
        "by_chapter": dict(sorted(chapter_counts.items())),
        "sample_exercise_ids": [
            d.metadata.get('exercise_id') 
            for d in all_docs 
            if d.metadata.get('exercise_id')
        ][:10],
        "sample_answer_ids": [
            d.metadata.get('answer_id') 
            for d in all_docs 
            if d.metadata.get('answer_id')
        ][:10]
    }
