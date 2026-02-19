from typing import List
from .rag_runtime import run_rag, vectorstore
from langchain_core.documents import Document

# -------------------------------------------------
#  Helper: Efficient metadata filtering
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
#  Exact Exercise Retrieval (fast hierarchical)
# -------------------------------------------------
def get_exercise(exercise_id: str, chapter: int | None = None, limit=5) -> List[Document]:
    """Get a specific exercise by its exact ID."""
    chapter = chapter or _parse_chapter_from_id(exercise_id)
    
    print(f"[DEBUG] Searching for exercise_id='{exercise_id}', chapter={chapter}")

    # Search all exercises first
    all_exercises = [
        d for d in vectorstore.docstore._dict.values()
        if d.metadata.get("type") == "exercise"
    ]
    
    print(f"[DEBUG] Total exercises in vectorstore: {len(all_exercises)}")
    
    # Filter by exercise_id
    results = [
        d for d in all_exercises
        if d.metadata.get("exercise_id") == exercise_id
    ]
    
    print(f"[DEBUG] Found {len(results)} matches for exercise {exercise_id}")
    
    if results:
        # Print first match metadata for debugging
        print(f"[DEBUG] First match metadata: {results[0].metadata}")
        print(f"[DEBUG] First match preview: {results[0].page_content[:200]}...")

    results.sort(key=lambda d: len(d.page_content or ""), reverse=True)
    return results[:limit]

# -------------------------------------------------
#  Exact Answer Retrieval (fast hierarchical)
# -------------------------------------------------

def get_answer(exercise_id: str, chapter: int | None = None, limit=5) -> List[Document]:
    """Get a specific answer by its exact exercise ID."""
    chapter = chapter or _parse_chapter_from_id(exercise_id)
    
    print(f"[DEBUG] Searching for answer to exercise_id='{exercise_id}', chapter={chapter}")

    # Get all answers
    all_answers = [
        d for d in vectorstore.docstore._dict.values()
        if d.metadata.get("section") == "answers_section"
        and d.metadata.get("type") == "answer"
    ]
    
    print(f"[DEBUG] Total answers in vectorstore: {len(all_answers)}")

    # Filter by answer_id (which matches exercise_id)
    results = [
        d for d in all_answers
        if d.metadata.get("answer_id") == exercise_id
    ]
    
    print(f"[DEBUG] Found {len(results)} matches for answer {exercise_id}")
    
    if results:
        print(f"[DEBUG] First match metadata: {results[0].metadata}")
        print(f"[DEBUG] First match preview: {results[0].page_content[:200]}...")

    results.sort(key=lambda d: len(d.page_content or ""), reverse=True)
    return results[:limit]



# -------------------------------------------------
# Exact Example Retrieval (fast hierarchical)
# -------------------------------------------------
def get_examples(example_id: str | None = None, chapter: int | None = None, limit=5) -> List[Document]:
    """Get examples either by ID or by chapter."""
    print(f"[DEBUG] Searching for example_id='{example_id}', chapter={chapter}")
    
    all_examples = [
        d for d in vectorstore.docstore._dict.values()
        if d.metadata.get("type") == "example"
    ]
    
    print(f"[DEBUG] Total examples in vectorstore: {len(all_examples)}")
    
    results = [
        d for d in all_examples
        if (not chapter or d.metadata.get("chapter") == chapter)
        and (not example_id or d.metadata.get("example_id") == example_id)
    ]

    print(f"[DEBUG] Found {len(results)} matching examples")

    results.sort(key=lambda d: len(d.page_content or ""), reverse=True)
    return results[:limit]

# -------------------------------------------------
#  Theory / Conceptual Retrieval (WITH similarity)
# -------------------------------------------------
def get_theory_concepts(query: str, chapter: int | None = None, limit=7) -> List[Document]:
    """Use similarity search across theory + example chunks."""
    print(f"[DEBUG] Theory search: query='{query[:50]}...', chapter={chapter}")
    
    # Build filter
    filter_query = {"type": {"$in": ["theory", "example"]}}
    if chapter:
        filter_query["chapter"] = chapter
    
    try:
        results = vectorstore.similarity_search(query, k=limit, filter=filter_query)
        print(f"[DEBUG] Found {len(results)} theory/example chunks")
        return results
    except Exception as e:
        print(f"[ERROR] Theory search failed: {e}")
        # Fallback: manual filtering
        all_docs = [
            d for d in vectorstore.docstore._dict.values()
            if d.metadata.get("type") in ["theory", "example"]
            and (not chapter or d.metadata.get("chapter") == chapter)
        ]
        return all_docs[:limit]
    
# -------------------------------------------------
# Hybrid Explanatory Tool (theory + example)
# -------------------------------------------------
def explain_with_context(query: str, chapter: int | None = None) -> str:
    """Use theory/examples from a chapter to explain a concept with LLM."""
    from .rag_runtime import run_rag
    
    docs = get_theory_concepts(query, chapter=chapter, limit=7)
    
    if not docs:
        return "No relevant theory or examples found in the textbook for this query."
    
    return run_rag(query, docs)


# Smart Search - Detects Query Type

def smart_search(query: str, chapter: int | None = None) -> dict:
    """
    Intelligently detect what the user is asking for and return appropriate results.
    Returns a dict with 'type', 'results', and 'formatted_text'.
    """
    import re
    
    query_lower = query.lower()
    
    # Pattern 1: Exercise request
    exercise_pattern = r'(?:exercise|problem|question)\s+(\d+\.\d+)'
    exercise_match = re.search(exercise_pattern, query_lower)
    if exercise_match:
        ex_id = exercise_match.group(1)
        docs = get_exercise(ex_id, chapter)
        return {
            'type': 'exercise',
            'exercise_id': ex_id,
            'results': docs,
            'formatted_text': "\n\n".join([d.page_content for d in docs]) if docs else f"Exercise {ex_id} not found."
        }
    
    # Pattern 2: Answer request
    answer_pattern = r'(?:answer|solution|solve).*?(\d+\.\d+)'
    answer_match = re.search(answer_pattern, query_lower)
    if answer_match:
        ex_id = answer_match.group(1)
        docs = get_answer(ex_id, chapter)
        return {
            'type': 'answer',
            'exercise_id': ex_id,
            'results': docs,
            'formatted_text': "\n\n".join([d.page_content for d in docs]) if docs else f"Answer to {ex_id} not found."
        }
    
    # Pattern 3: Example request
    example_pattern = r'example\s+(\d+(?:\.\d+)*)'
    example_match = re.search(example_pattern, query_lower)
    if example_match:
        ex_id = example_match.group(1)
        docs = get_examples(ex_id, chapter)
        return {
            'type': 'example',
            'example_id': ex_id,
            'results': docs,
            'formatted_text': "\n\n".join([d.page_content for d in docs]) if docs else f"Example {ex_id} not found."
        }
    
    # Pattern 4: Conceptual/Theory question
    docs = get_theory_concepts(query, chapter, limit=7)
    return {
        'type': 'theory',
        'results': docs,
        'formatted_text': "\n\n".join([d.page_content for d in docs]) if docs else "No relevant theory found."
    }
