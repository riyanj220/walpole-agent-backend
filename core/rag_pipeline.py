from .rag_chain import run_rag, vectorstore
from .formulas import FORMULAS
import re

EXERCISE_REF_RE = re.compile(r"\b(\d+\.\d+)\b")  # matches 5.10, 11.3, etc.

def get_docs_by_exercise_id(exercise_id: str):
    docs = [
            d for d in vectorstore.docstore._dict.values()
            if d.metadata.get("type") == "exercise"
            and d.metadata.get("exercise_id") == exercise_id
        ]   
    # If multiple chunks exist for the same exercise, pick the largest text (usually full prompt)
    docs.sort(key=lambda d: len(d.page_content or ""), reverse=True)
    return docs


def get_answer_by_exercise_id(exercise_id: str):
    docs = [
        d for d in vectorstore.docstore._dict.values()
        if d.metadata.get("type") == "answer"
        and d.metadata.get("exercise_id") == exercise_id
    ]
    docs.sort(key=lambda d: len(d.page_content or ""), reverse=True)
    return docs 



def ask_pipeline(query: str, params: dict | None = None):
    q = query.lower()

    # --- Formula path ---
    for name, f in FORMULAS.items():
        if all(word in q for word in f["keywords"]):
            result = None
            try:
                if params:
                    result = f["func"](**params)
            except Exception:
                pass

            return {
                "mode": "formula",
                "query": query,
                "result": float(result) if result is not None else None,
                "formula": f["latex"],
                "steps": f.get("steps", []),
                "explanation": run_rag(query, []),  # explanation from LLM, no docs
            }


    ANSWER_WORDS = ("answer", "solution", "key", "solve part", "final value")
    exercise_match = EXERCISE_REF_RE.search(q)

    # --- exact answer lookup first ---
    if exercise_match and any(w in q for w in ANSWER_WORDS):
        exercise_id = exercise_match.group(1)
        docs = get_answer_by_exercise_id(exercise_id)
        if docs:
            # You can return raw answer text or let LLM format it:
            return {
                "mode": "answer",
                "query": query,
                "exercise_id": exercise_id,
                "context": docs[0].page_content,
                "answer": docs[0].page_content,
            }
        else:
            # fall back to solving from the exercise statement if available
            ex_docs = get_docs_by_exercise_id(exercise_id)
            if ex_docs:
                return {
                    "mode": "rag",
                    "query": query,
                    "exercise_id": exercise_id,
                    "context": ex_docs[0].page_content,
                    "answer": run_rag(query, [ex_docs[0]]),
                }
            return {
                "mode": "answer",
                "query": query,
                "exercise_id": exercise_id,
                "answer": "Answer not found in the book context.",
            }

    # --- exercise lookup second ---
    if "exercise" in q and exercise_match:
        exercise_id = exercise_match.group(1)
        docs = get_docs_by_exercise_id(exercise_id)
        if docs:
            return {
                "mode": "rag",
                "query": query,
                "exercise_id": exercise_id,
                "context": docs[0].page_content,
                "answer": run_rag(query, [docs[0]]),
            }
        return {
            "mode": "rag",
            "query": query,
            "exercise_id": exercise_id,
            "answer": "Exercise not found in the book context.",
        }

    return {
        "mode": "rag",
        "query": query,
        "filter_type": None
    }
