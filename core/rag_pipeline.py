"""
Main RAG Pipeline Orchestrator
Handles routing between agent mode and direct mode based on query complexity.
"""

from .rag_agent import ask_agent, ask_direct
from .rag_tools import smart_search
import re


def detect_query_complexity(query: str) -> str:
    """
    Determine if query needs agent reasoning or can be handled directly.
    
    Returns:
        'simple' - Direct retrieval (exercise, answer, example by ID)
        'complex' - Needs agent reasoning (multi-step, conceptual)
    """
    query_lower = query.lower()
    
    # Simple patterns: Direct ID-based retrieval
    simple_patterns = [
        r'exercise\s+\d+\.\d+',
        r'problem\s+\d+\.\d+',
        r'question\s+\d+\.\d+',
        r'answer.*?\d+\.\d+',
        r'solution.*?\d+\.\d+',
        r'example\s+\d+(?:\.\d+)*'
    ]
    
    for pattern in simple_patterns:
        if re.search(pattern, query_lower):
            return 'simple'
    
    # Complex patterns: Need reasoning
    complex_indicators = [
        'explain', 'how', 'why', 'compare', 'difference',
        'relationship', 'derive', 'prove', 'show that',
        'step by step', 'detail', 'understand'
    ]
    
    if any(indicator in query_lower for indicator in complex_indicators):
        return 'complex'
    
    # Default to complex for safety
    return 'complex'


def ask_pipeline(query: str, params: dict = None) -> dict:
    """
    Main pipeline entry point.
    
    Args:
        query: User's question
        params: Optional parameters
            - chapter (int): Chapter number filter
            - mode (str): 'agent' or 'direct' to force specific mode
            - max_results (int): Maximum number of results
            
    Returns:
        dict with:
            - answer (str): The generated answer
            - mode (str): Which mode was used ('agent' or 'direct')
            - metadata (dict): Additional information
            - sources (list): Source documents used
    """
    params = params or {}
    chapter = params.get('chapter')
    forced_mode = params.get('mode')
    
    # Determine mode
    if forced_mode:
        mode = forced_mode
    else:
        complexity = detect_query_complexity(query)
        mode = 'direct' if complexity == 'simple' else 'agent'
    
    print(f"[Pipeline] Mode: {mode} | Chapter: {chapter} | Query: {query[:50]}...")
    
    # Execute based on mode
    if mode == 'direct':
        result = ask_direct(query, chapter)
        return {
            "answer": result['answer'],
            "mode": "direct",
            "metadata": result['metadata'],
            "query_type": result.get('type', 'unknown')
        }
    else:
        result = ask_agent(query, chapter)
        return {
            "answer": result['answer'],
            "mode": "agent",
            "metadata": result['metadata'],
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