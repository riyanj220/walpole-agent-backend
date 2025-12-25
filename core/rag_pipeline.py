"""
Main RAG Pipeline Orchestrator
Uses an LLM Router to intelligently distinguish between:
1. General Chat (Greetings, feelings, small talk)
2. Direct Content (Exercises, definitions, simple questions)
3. Agent Reasoning (Complex comparisons, multi-step logic)
"""

import os
from decouple import config

os.environ["LANGCHAIN_TRACING_V2"] = config("LANGSMITH_TRACING_V2", default="false")
os.environ["LANGCHAIN_ENDPOINT"] = config("LANGSMITH_ENDPOINT", default="https://eu.api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = config("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = config("LANGSMITH_PROJECT", default="pr-brief-electrocardiogram-6")


from .rag_agent import ask_agent, ask_direct
from .rag_runtime import run_general_chat, llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import re
from langsmith import traceable
from langsmith import Client

client = Client()

try:
    if not client.has_project(project_name="pr-brief-electrocardiogram-6"):
        client.create_project(project_name="pr-brief-electrocardiogram-6")
        print("Successfully created LangSmith project.")
except Exception as e:
    print(f"Project check failed: {e}")


def semantic_router(query: str, chat_history: list=[]) -> str:
    """
    Uses the LLM to decide the intent of the user.
    Returns one of: 'general', 'direct', 'agent'
    """
    try:
        context_str = ""
        if chat_history:
            last_role, last_msg = chat_history[-1]
            context_str = f"\nPREVIOUS MESSAGE ({last_role}): {last_msg[:100]}...\n"

        # This prompt teaches the LLM how to route traffic
        router_prompt = ChatPromptTemplate.from_template("""
        You are an intent classifier for a Probability & Statistics Tutor Bot.
        Analyze the user's input and classify it into one of three categories:
        
        1. "general": 
           - Casual conversation ("hi", "how are you", "cool", "thanks")
           - Emotional responses ("I'm stressed", "this is hard", "it's great")
           - General study advice ("how should I study?")
           - Anything NOT related to specific probability/statistics content.
           - BUT: If the user is asking to explain the previous answer, it is NOT general.
           - IMPORTANT: If the user asks for "an example", "more details", "why", or "how", it is NOT general.                                                            

        2. "direct":
           - Questions asking for specific exercises ("exercise 6.9", "6.14")
           - Questions asking for answers ("answer to 2.3")
           - "How to solve" specific problems.
           - Simple definitions ("what is variance?", "define mean")
           - Follow-up questions asking to elaborate on the previous math topic.                                                                    

        3. "agent":
           - Complex questions requiring reasoning.
           - Comparisons ("difference between X and Y")
           - Relationships ("how does sample size affect error?")
        
        CONTEXT FROM HISTORY: {context_str}
                                                                  
        USER INPUT: "{query}"
        
        Return a JSON object with a single key "route".
        Example: {{"route": "general"}}

        STRICT INSTRUCTIONS:
        - Do NOT output any Python code or explanations.
        - Do NOT output markdown ticks (```json).
        - Output ONLY the raw JSON object.
        """)

        # Create chain: Prompt -> LLM -> JSON Parser
        chain = router_prompt | llm | JsonOutputParser()
        
        # Run the router
        result = chain.invoke({"query": query, "context_str": context_str})
        route = result.get("route", "direct")
        
        print(f"[Router] Analyzed intent: '{query}' -> {route.upper()}")
        return route

    except Exception as e:
        print(f"[Router Error] {e}. Falling back to Regex.")
        return fallback_regex_router(query)


def fallback_regex_router(query: str) -> str:
    """
    Backup logic in case the LLM Router fails (network error, etc).
    """
    query_lower = query.lower()
    
    # 1. Check for ID (Strongest Signal)
    if re.search(r'\d+\.\d+', query_lower):
        return 'direct'
        
    # 2. Check for Chat keywords
    chat_triggers = ['hi', 'hello', 'thanks', 'great', 'good', 'bye', 'stress']
    if any(x in query_lower for x in chat_triggers):
        return 'general'
        
    # 3. Check for Agent keywords
    agent_triggers = ['compare', 'difference', ' vs ']
    if any(x in query_lower for x in agent_triggers):
        return 'agent'
        
    return 'direct'

@traceable(project_name="pr-brief-electrocardiogram-6")
def ask_pipeline(query: str, params: dict = None, chat_history: list = []) -> dict:
    """
    Main pipeline entry point.
    """
    params = params or {}
    chapter = params.get('chapter')
    forced_mode = params.get('mode')
    
    # 1. Determine Route (LLM Decision)
    if forced_mode:
        mode = forced_mode
    else:
        mode = semantic_router(query)
    
    print(f"[Pipeline] Mode: {mode} | Chapter: {chapter} | Query: {query[:50]}...")
    
    # 2. Execute Route
    
    # --- ROUTE A: GENERAL CHAT (No DB) ---
    if mode == 'general':
        answer = run_general_chat(query, chat_history)
        return {
            "answer": answer,
            "mode": "general_chat",
            "metadata": {"success": True}
        }

    # --- ROUTE B: DIRECT RAG (Vector Search) ---
    elif mode == 'direct':
        result = ask_direct(query, chapter, chat_history)
        return {
            "answer": result['answer'],
            "mode": "direct",
            "metadata": result.get('metadata', {}),
            "query_type": result.get('type', 'unknown')
        }
        
    # --- ROUTE C: AGENT RAG (Reasoning) ---
    else:
        result = ask_agent(query, chapter, chat_history)
        return {
            "answer": result['answer'],
            "mode": "agent",
            "metadata": result.get('metadata', {}),
            "reasoning_steps": len(result.get('steps', []))
        }

def batch_ask(queries: list, params: dict = None) -> list:
    """
    Process multiple queries in batch.
    
    Args:
        queries: List of query strings
        params: Optional parameters applied to all queries
        
    Returns:
        List of results from ask_pipeline
    """
    results = []
    for query in queries:
        result = ask_pipeline(query, params)
        results.append(result)
    return results


def get_chapter_summary(chapter: int, limit: int = 10) -> dict:
    """
    Get a summary of available content for a specific chapter.
    
    Args:
        chapter: Chapter number
        limit: Max items per category
        
    Returns:
        dict with exercises, examples, and theory sections
    """
    from .rag_runtime import vectorstore
    
    # Get all documents for this chapter
    chapter_docs = [
        d for d in vectorstore.docstore._dict.values()
        if d.metadata.get('chapter') == chapter
    ]
    
    # Categorize
    exercises = [d for d in chapter_docs if d.metadata.get('type') == 'exercise'][:limit]
    examples = [d for d in chapter_docs if d.metadata.get('type') == 'example'][:limit]
    theory = [d for d in chapter_docs if d.metadata.get('type') == 'theory'][:limit]
    answers = [d for d in chapter_docs if d.metadata.get('type') == 'answer'][:limit]
    
    return {
        "chapter": chapter,
        "summary": {
            "total_chunks": len(chapter_docs),
            "exercises": len(exercises),
            "examples": len(examples),
            "theory_sections": len(theory),
            "answers": len(answers)
        },
        "exercise_ids": [d.metadata.get('exercise_id') for d in exercises if d.metadata.get('exercise_id')],
        "example_ids": [d.metadata.get('example_id') for d in examples if d.metadata.get('example_id')],
        "available_answers": [d.metadata.get('answer_id') for d in answers if d.metadata.get('answer_id')]
    }


def health_check() -> dict:
    """
    Check if the RAG system is properly initialized.
    
    Returns:
        dict with system status
    """
    from .rag_runtime import vectorstore
    
    try:
        total_docs = len(vectorstore.docstore._dict)
        
        # Count by type
        type_counts = {}
        for doc in vectorstore.docstore._dict.values():
            doc_type = doc.metadata.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return {
            "status": "healthy",
            "total_documents": total_docs,
            "document_types": type_counts,
            "vectorstore": "loaded"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }