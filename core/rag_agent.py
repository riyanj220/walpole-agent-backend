from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from .rag_tools import (
    get_exercise, 
    get_answer, 
    explain_with_context, 
    get_theory_concepts,
    get_examples,
    smart_search
)
from .rag_runtime import run_rag,llm
import re

# =====================================================
# Tool Definitions
# =====================================================

def exercise_tool(query: str) -> str:
    """Extract exercise ID and retrieve exercise content."""
    # Extract exercise ID from query
    match = re.search(r'(\d+\.\d+)', query)
    if not match:
        return "Please specify an exercise ID (e.g., '6.14')"
    
    exercise_id = match.group(1)
    docs = get_exercise(exercise_id)
    
    if not docs:
        return f"Exercise {exercise_id} not found in the textbook."
    
    return "\n\n".join([d.page_content for d in docs])


def answer_tool(query: str) -> str:
    """Extract exercise ID and retrieve answer content."""
    match = re.search(r'(\d+\.\d+)', query)
    if not match:
        return "Please specify an exercise ID (e.g., '6.14')"
    
    exercise_id = match.group(1)
    docs = get_answer(exercise_id)
    
    if not docs:
        return f"Answer to exercise {exercise_id} not found in the textbook."
    
    return "\n\n".join([d.page_content for d in docs])


def example_tool(query: str) -> str:
    """Extract example ID and retrieve example content."""
    match = re.search(r'(\d+(?:\.\d+)*)', query)
    if not match:
        return "Please specify an example ID (e.g., '5.3')"
    
    example_id = match.group(1)
    docs = get_examples(example_id)
    
    if not docs:
        return f"Example {example_id} not found in the textbook."
    
    return "\n\n".join([d.page_content for d in docs])


def theory_tool(query: str) -> str:
    """Retrieve relevant theory and conceptual content."""
    docs = get_theory_concepts(query, limit=5)
    
    if not docs:
        return "No relevant theory found for this query."
    
    context = "\n\n".join([d.page_content for d in docs])
    return run_rag(query, docs)


def explain_tool(query: str) -> str:
    """Provide detailed explanation with theory and examples."""
    return explain_with_context(query)


# =====================================================
#  Tools List
# =====================================================
tools = [
    Tool(
        name="GetExercise",
        func=exercise_tool,
        description=(
            "Use this to retrieve a specific exercise/problem text. "
            "Input MUST be the exercise number (e.g., '6.14'). "
            "Returns the full exercise text from the textbook."
        )
    ),
    Tool(
        name="GetAnswer",
        func=answer_tool,
        description=(
            "Use this to retrieve the answer/solution to an exercise. "
            "Input MUST be the exercise number (e.g., '6.14'). "
            "Returns the answer from the textbook solutions section."
        )
    ),
    Tool(
        name="GetExample",
        func=example_tool,
        description=(
            "Use this to retrieve a worked example from the textbook. "
            "Input MUST be the example number (e.g., '5.3'). "
            "Returns the full example with solution."
        )
    ),
    Tool(
        name="SearchTheory",
        func=theory_tool,
        description=(
            "Use this to search for theoretical concepts, definitions, or explanations. "
            "Input should be the concept or topic (e.g., 'variance', 'Bayes theorem'). "
            "Returns relevant theory sections from the textbook."
        )
    )
]


# =====================================================
# Agent Prompt Template
# =====================================================
agent_prompt = PromptTemplate.from_template("""You are a probability and statistics textbook assistant. Your job is to help students by retrieving specific content from the textbook.

AVAILABLE TOOLS:
{tools}

TOOL NAMES: {tool_names}

IMPORTANT RULES:
1. For exercise questions → Use GetExercise with just the number (e.g., "6.14")
2. For answer questions → Use GetAnswer with just the number (e.g., "6.14")
3. For examples → Use GetExample with just the number (e.g., "5.3")
4. For concepts/theory → Use SearchTheory with the concept name
5. ALWAYS use tools - do NOT try to answer from memory.
6. **After you have gathered all the information you need using the available tools, you MUST stop calling tools and provide a "Final Answer".**
7. **DO NOT invent new tool names.** Only use the tools provided: {tool_names}.
8. If a tool returns "ERROR:", try a different approach or inform the user.
9. Keep your final answer concise and based ONLY on tool results.

FORMAT TO FOLLOW:
Question: the input question
Thought: what should I do?
Action: one of [{tool_names}]
Action Input: the input for the tool
Observation: the result from the tool
... (repeat Thought/Action/Observation if needed)
Thought: I now have all the information needed to answer the question.
Final Answer: [your answer based on observations]

BEGIN!

Question: {input}
Thought: {agent_scratchpad}""")

# =====================================================
# Create Agent
# =====================================================
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=agent_prompt
)

# Wrap the agent in an executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5, 
    max_execution_time=30,  # 30 second timeout
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    early_stopping_method="force"    
)

# =====================================================
#  Main Agent Interface
# =====================================================
def ask_agent(query: str, chapter: int = None) -> dict:
    """
    Main interface to ask the AI Agent.
    """
    try:
        enhanced_query = query
        if chapter:
            enhanced_query = f"[From Chapter {chapter}] {query}"
        
        print(f"[AGENT] Processing: {enhanced_query}")
        
        # Execute agent
        result = agent_executor.invoke({"input": enhanced_query})
        
        tools_used = [
            step[0].tool for step in result.get("intermediate_steps", [])
            if hasattr(step[0], 'tool')
        ]
        
        return {
            "answer": result["output"],
            "steps": result.get("intermediate_steps", []),
            "metadata": {
                "chapter": chapter,
                "query": query,
                "tools_used": tools_used,
                "success": True
            }
        }
    except Exception as e:
        print(f"[AGENT ERROR] {str(e)}")
        return {
            "answer": f"I encountered an error processing your query. Please try rephrasing or use direct mode. Error: {str(e)}",
            "steps": [],
            "metadata": {
                "chapter": chapter,
                "query": query,
                "error": str(e),
                "success": False
            }
        }


# =====================================================
#  Direct Search (IMPROVED)
# =====================================================
def ask_direct(query: str, chapter: int = None) -> dict:
    """
    Direct retrieval without agent reasoning.
    Much faster for simple ID-based queries.
    """
    try:
        print(f"[DIRECT] Processing: {query}")
        result = smart_search(query, chapter)
        
        # --- NEW LOGIC: Detect "how-to-solve" queries ---
        query_lower = query.lower()
        is_solve_query = (
            result['type'] in ['exercise', 'answer'] and
            any(k in query_lower for k in ['how to', 'explain', 'solve', 'step by step', 'steps to'])
        )

        if is_solve_query:
            print("[DIRECT] Detected 'how-to-solve' query. Gathering context...")
            ex_id = result.get('exercise_id')
            if not ex_id:
                 # Fallback if smart_search didn't find an ID (e.g. "how to solve this [pasted text]")
                return run_rag(query, result['results'])

            # 1. Get Exercise
            exercise_docs = get_exercise(ex_id, chapter)
            exercise_text = exercise_docs[0].page_content if exercise_docs else "Exercise text not found."
            
            # 2. Get Answer
            answer_docs = get_answer(ex_id, chapter) # Get fresh in case result['type'] was 'exercise'
            
            # 3. Get relevant Theory
            # We use the exercise text itself to find related theory
            theory_docs = get_theory_concepts(exercise_text, chapter, limit=3)
            
            # 4. Bundle all docs
            all_docs = exercise_docs + answer_docs + theory_docs
            if not all_docs:
                return { "answer": f"I found exercise {ex_id}, but I couldn't find the text, answer, or related theory to explain it.", "type": "error", "metadata": { "chapter": chapter, "query": query, "success": False } }

            # 5. Create a new, specific query for the LLM
            solve_query = f"""
            A student is asking how to solve an exercise. Your job is to provide a step-by-step explanation.
            Use the following context:
            1. The EXERCISE.
            2. The FINAL ANSWER (if available).
            3. Related THEORY from the textbook.

            **Student's Question:** {query}
            **Exercise {ex_id}:** {exercise_text}

            Provide a clear, pedagogical, step-by-step explanation of how to arrive at the final answer.
            Base your explanation ONLY on the provided context.
            """
            
            # 6. Call run_rag with the new prompt and bundled context
            answer = run_rag(solve_query, all_docs)
            
            return {
                "answer": answer,
                "type": "explanation",
                "metadata": {
                    "chapter": chapter,
                    "query": query,
                    "exercise_id": ex_id,
                    "num_results": len(all_docs),
                    "success": True
                }
            }

        # If we have results, format them nicely (Original Logic)
        if result['results']:
            # For simple exercises/answers, return raw content
            if result['type'] in ['exercise', 'answer', 'example']:
                answer = result['formatted_text']
            # For theory, use LLM to synthesize
            else:
                answer = run_rag(query, result['results'])
        else:
            answer = result['formatted_text']
        
        return {
            "answer": answer,
            "type": result['type'],
            "metadata": {
                "chapter": chapter,
                "query": query,
                "num_results": len(result['results']),
                "success": len(result['results']) > 0
            }
        }
    except Exception as e:
        print(f"[DIRECT ERROR] {str(e)}")
        return {
            "answer": f"Error processing query: {str(e)}",
            "type": "error",
            "metadata": {
                "chapter": chapter,
                "query": query,
                "error": str(e),
                "success": False
            }
        }