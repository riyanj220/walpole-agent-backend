from typing import List
from .rag_runtime import run_rag, vectorstore
from langchain_core.documents import Document

# -------------------------------------------------
# ⚙️ Helper: Efficient metadata filtering
# -------------------------------------------------
def _filter_docs(filter_dict: dict, sort_by_len=True, limit=None):
    """Internal helper to filter documents by exact metadata."""
    results = [
        d for d in vectorstore.docstore._dict.values()
        if all(d.metadata.get(k) == v for k, v in filter_dict.items())
    ]
    if sort_by_len:
        results.sort(key=lambda d: len(d.page_content or ""), reverse=True)
    return results[:limit] if limit else results


def _parse_chapter_from_id(ex_id: str | None):
    """Extract chapter number from exercise_id or answer_id (e.g., '18.14' -> 18)."""
    try:
        return int(ex_id.split(".")[0])
    except Exception:
        return None


# -------------------------------------------------
# 🎯 1️⃣ Exact Exercise Retrieval (fast hierarchical)
# -------------------------------------------------
def get_exercise(exercise_id: str, chapter: int | None = None, limit=3) -> List[Document]:
    """Get a specific exercise by its exact ID, using chapter narrowing for faster search."""
    chapter = chapter or _parse_chapter_from_id(exercise_id)

    results = [
        d for d in vectorstore.docstore._dict.values()
        if d.metadata.get("type") == "exercise"
        and (not chapter or d.metadata.get("chapter") == chapter)
        and d.metadata.get("exercise_id") == exercise_id
    ]

    if not results:
        print(f"[⚠️] Exercise {exercise_id} not found (chapter={chapter}).")
        return []

    results.sort(key=lambda d: len(d.page_content or ""), reverse=True)
    return results[:limit]


# -------------------------------------------------
# 🧩 2️⃣ Exact Answer Retrieval (fast hierarchical)
# -------------------------------------------------
def get_answer(exercise_id: str, chapter: int | None = None, limit=3) -> List[Document]:
    """Get a specific answer by its exact exercise ID."""
    chapter = chapter or _parse_chapter_from_id(exercise_id)

    answer_candidates = [
        d for d in vectorstore.docstore._dict.values()
        if d.metadata.get("section") == "answers_section"
    ]

    if chapter:
        answer_candidates = [d for d in answer_candidates if d.metadata.get("chapter") == chapter]

    results = [
        d for d in answer_candidates
        if d.metadata.get("type") == "answer"
        and d.metadata.get("answer_id") == exercise_id
    ]

    if not results:
        print(f"[⚠️] Answer {exercise_id} not found (chapter={chapter}).")
        return []

    results.sort(key=lambda d: len(d.page_content or ""), reverse=True)
    return results[:limit]


# -------------------------------------------------
# 📘 3️⃣ Exact Example Retrieval (fast hierarchical)
# -------------------------------------------------
def get_examples(example_id: str | None = None, chapter: int | None = None, limit=5) -> List[Document]:
    """Get examples either by ID or by chapter."""
    results = [
        d for d in vectorstore.docstore._dict.values()
        if d.metadata.get("type") == "example"
        and (not chapter or d.metadata.get("chapter") == chapter)
        and (not example_id or d.metadata.get("example_id") == example_id)
    ]

    if not results:
        print(f"[⚠️] Example {example_id or ''} not found (chapter={chapter}).")
        return []

    results.sort(key=lambda d: len(d.page_content or ""), reverse=True)
    return results[:limit]


# -------------------------------------------------
# 📗 4️⃣ Theory / Conceptual Retrieval (WITH similarity)
# -------------------------------------------------
def get_theory_concepts(query: str, chapter: int | None = None, limit=5) -> List[Document]:
    """Use similarity search across theory + example chunks, optionally limited by chapter."""
    filter_query = {"type": {"$in": ["theory", "example"]}}
    if chapter:
        filter_query["chapter"] = chapter

    return vectorstore.similarity_search(query, k=limit, filter=filter_query)


# -------------------------------------------------
# 💡 5️⃣ Hybrid Explanatory Tool (theory + example)
# -------------------------------------------------
def explain_with_context(query: str, chapter: int | None = None) -> str:
    """Use theory/examples from a chapter to explain a concept with LLM."""
    docs = get_theory_concepts(query, chapter=chapter, limit=5)
    return run_rag(query, docs)


# Smart Search - Detects Query Type

def smart_search(query: str, chapter: int | None = None) -> dict:
    """
    Intelligently detect what the user is asking for and return appropriate results.
    Returns a dict with 'type', 'results', and 'formatted_text'.
    """
    import re
    
    query_lower = query.lower()
    
    # Pattern 1: Exercise request (e.g., "exercise 6.14", "problem 12.5", "question 3.7")
    exercise_pattern = r'(?:exercise|problem|question)\s+(\d+\.\d+)'
    exercise_match = re.search(exercise_pattern, query_lower)
    if exercise_match:
        ex_id = exercise_match.group(1)
        docs = get_exercise(ex_id, chapter)
        return {
            'type': 'exercise',
            'exercise_id': ex_id,
            'results': docs,
            'formatted_text': "\n\n".join([d.page_content for d in docs]) if docs else "Exercise not found."
        }
    
    # Pattern 2: Answer request (e.g., "answer to 6.14", "solution for 12.5", "solve 3.7")
    answer_pattern = r'(?:answer|solution|solve).*?(\d+\.\d+)'
    answer_match = re.search(answer_pattern, query_lower)
    if answer_match:
        ex_id = answer_match.group(1)
        docs = get_answer(ex_id, chapter)
        return {
            'type': 'answer',
            'exercise_id': ex_id,
            'results': docs,
            'formatted_text': "\n\n".join([d.page_content for d in docs]) if docs else "Answer not found."
        }
    
    # Pattern 3: Example request (e.g., "example 5.3", "show example from chapter 7")
    example_pattern = r'example\s+(\d+(?:\.\d+)*)'
    example_match = re.search(example_pattern, query_lower)
    if example_match:
        ex_id = example_match.group(1)
        docs = get_examples(ex_id, chapter)
        return {
            'type': 'example',
            'example_id': ex_id,
            'results': docs,
            'formatted_text': "\n\n".join([d.page_content for d in docs]) if docs else "Example not found."
        }
    
    # Pattern 4: Conceptual/Theory question (default)
    docs = get_theory_concepts(query, chapter)
    return {
        'type': 'theory',
        'results': docs,
        'formatted_text': "\n\n".join([d.page_content for d in docs]) if docs else "No relevant theory found."
    }
