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
            "This tool returns a COMPLETE, FINAL ANSWER. You should output this answer directly."
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
6. **SPECIAL RULE: The 'SearchTheory' tool provides a complete answer. After you use 'SearchTheory', your next step MUST be to output a 'Final Answer' with the text from the 'Observation'. Do not loop.**
7. **After you have gathered all the information you need using other tools, you MUST stop calling tools and provide a "Final Answer".**
8. **DO NOT invent new tool names.** Only use the tools provided: {tool_names}.
9. If a tool returns "ERROR:", try a different approach or inform the user.
10. Keep your final answer concise and based ONLY on tool results.

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
    Optimized for Exercise/Answer retrieval.
    """
    try:
        print(f"[DIRECT] Processing: {query}")
        
        # 1. Try to find an Exercise ID immediately (Regex is faster/more accurate than semantic search)
        # Matches "6.9", "Exercise 6.9", "6.14", etc.
        id_match = re.search(r'(\d+\.\d+)', query)
        
        # --- PATH A: ID DETECTED (The "Smart" Path) ---
        if id_match:
            ex_id = id_match.group(1)
            print(f"[DIRECT] Detected ID {ex_id}. Entering targeted retrieval mode.")
            
            # A. Get Exercise Text
            exercise_docs = get_exercise(ex_id, chapter)
            exercise_text = exercise_docs[0].page_content if exercise_docs else f"Text for exercise {ex_id} not found."
            
            # B. Get Answer Key (Crucial Step)
            answer_docs = get_answer(ex_id, chapter)
            answer_text = answer_docs[0].page_content if answer_docs else "Answer key not found in database."
            
            # C. Get Related Theory (Context)
            theory_docs = get_theory_concepts(exercise_text, chapter, limit=2)
            
            # D. Combine Context
            all_docs = exercise_docs + answer_docs + theory_docs
            
            # E. Prompt the LLM specifically to look for the answer
            solve_query = f"""
            You are a helpful tutor. A student is asking about Exercise {ex_id}.
            
            User Query: "{query}"
            
            Here is the data from the textbook:
            1. [THE EXERCISE PROBLEM]:
            {exercise_text}
            
            2. [THE OFFICIAL ANSWER KEY]:
            {answer_text}
            
            3. [RELEVANT THEORY]:
            {chr(10).join([d.page_content for d in theory_docs])}
            
            INSTRUCTIONS:
            - If the user asked for the **answer**, provide the 'Official Answer Key' clearly.
            - If the user asked **how to solve it**, provide a step-by-step explanation using the Theory.
            - If the answer key says "Not found", admit that you don't have the official final number, but explain the method.
            """
            
            answer = run_rag(solve_query, all_docs)
            
            return {
                "answer": answer,
                "type": "targeted_retrieval",
                "metadata": {
                    "chapter": chapter,
                    "query": query,
                    "exercise_id": ex_id,
                    "found_answer": bool(answer_docs),
                    "success": True
                }
            }

        # --- PATH B: NO ID (General Semantic Search) ---
        # This handles "What is Bayes theorem?" etc.
        else:
            print("[DIRECT] No ID detected. Running semantic search.")
            result = smart_search(query, chapter)
            
            if result['results']:
                answer = run_rag(query, result['results'])
            else:
                answer = "I couldn't find relevant information in the textbook for that query."
            
            return {
                "answer": answer,
                "type": result.get('type', 'general'),
                "metadata": {
                    "chapter": chapter,
                    "query": query,
                    "num_results": len(result.get('results', [])),
                    "success": len(result.get('results', [])) > 0
                }
            }

    except Exception as e:
        print(f"[DIRECT ERROR] {str(e)}")
        return {
            "answer": f"Error processing query: {str(e)}",
            "type": "error",
            "metadata": {"error": str(e), "success": False}
        }
    

