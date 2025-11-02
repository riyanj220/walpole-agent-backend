from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from .rag_tools import (
    get_exercise, 
    get_answer, 
    explain_with_context, 
    get_theory_concepts,
    get_examples,
    smart_search
)
from .rag_runtime import run_rag
from decouple import config
import re

# =====================================================
# 🔹 Configuration
# =====================================================
OLLAMA_BASE_URL = config("OLLAMA_BASE_URL", default="http://localhost:11434")
MODEL_NAME = "llama3.2:1b"

# =====================================================
# 🔹 LLM Setup
# =====================================================
llm = OllamaLLM(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0
)

# =====================================================
# 🔹 Tool Definitions
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
# 🔹 Tools List
# =====================================================
tools = [
    Tool(
        name="GetExercise",
        func=exercise_tool,
        description=(
            "Use this when the user asks for a specific exercise or problem. "
            "Input should contain the exercise ID (e.g., '6.14' or 'exercise 12.5'). "
            "Examples: 'Show me exercise 6.14', 'What is problem 12.5?'"
        )
    ),
    Tool(
        name="GetAnswer",
        func=answer_tool,
        description=(
            "Use this when the user asks for the answer or solution to a specific exercise. "
            "Input should contain the exercise ID. "
            "Examples: 'Answer to 6.14', 'What's the solution for 12.5?', 'Solve exercise 3.7'"
        )
    ),
    Tool(
        name="GetExample",
        func=example_tool,
        description=(
            "Use this when the user asks for a specific example from the textbook. "
            "Input should contain the example ID. "
            "Examples: 'Show me example 5.3', 'What is example 7.2?'"
        )
    ),
    Tool(
        name="GetTheory",
        func=theory_tool,
        description=(
            "Use this for conceptual or theoretical questions about probability topics. "
            "Input should be the theoretical concept or question. "
            "Examples: 'What is variance?', 'Explain Bayes theorem', 'What is posterior distribution?'"
        )
    ),
    Tool(
        name="ExplainConcept",
        func=explain_tool,
        description=(
            "Use this when the user needs a detailed explanation combining theory and examples. "
            "Input should be the concept or topic to explain. "
            "Examples: 'Explain conditional probability with examples', 'How does the central limit theorem work?'"
        )
    )
]

# =====================================================
# 🔹 Agent Prompt Template
# =====================================================
agent_prompt = PromptTemplate.from_template("""You are a helpful probability and statistics tutor assistant. 
Your job is to help students by using the available tools to answer their questions.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Guidelines:
- For exercise questions (e.g., "exercise 6.14"), use GetExercise
- For answer questions (e.g., "answer to 6.14"), use GetAnswer
- For example questions (e.g., "example 5.3"), use GetExample
- For conceptual questions (e.g., "what is variance"), use GetTheory
- For detailed explanations, use ExplainConcept
- Always provide clear and complete responses
- If content is not found, inform the user politely

Question: {input}
Thought: {agent_scratchpad}""")

# =====================================================
# 🔹 Create Agent
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
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

# =====================================================
# 🔹 Main Agent Interface
# =====================================================
def ask_agent(query: str, chapter: int = None) -> dict:
    """
    Main interface to ask the AI Agent.
    
    Args:
        query: User's question
        chapter: Optional chapter number to narrow search
        
    Returns:
        dict with 'answer', 'steps', and 'metadata'
    """
    try:
        # Add chapter context to query if provided
        enhanced_query = query
        if chapter:
            enhanced_query = f"[Chapter {chapter}] {query}"
        
        # Execute agent
        result = agent_executor.invoke({"input": enhanced_query})
        
        return {
            "answer": result["output"],
            "steps": result.get("intermediate_steps", []),
            "metadata": {
                "chapter": chapter,
                "query": query,
                "tools_used": [step[0].tool for step in result.get("intermediate_steps", [])]
            }
        }
    except Exception as e:
        return {
            "answer": f"Error processing query: {str(e)}",
            "steps": [],
            "metadata": {"error": str(e)}
        }


# =====================================================
# 🔹 Fallback: Direct Smart Search (No Agent)
# =====================================================
def ask_direct(query: str, chapter: int = None) -> dict:
    """
    Fallback method: Use smart_search to directly retrieve content without agent reasoning.
    Useful for simple queries or when agent is too slow.
    
    Args:
        query: User's question
        chapter: Optional chapter number
        
    Returns:
        dict with 'answer', 'type', and 'metadata'
    """
    try:
        result = smart_search(query, chapter)
        
        # If we have results, use RAG to generate answer
        if result['results']:
            answer = run_rag(query, result['results'])
        else:
            answer = result['formatted_text']
        
        return {
            "answer": answer,
            "type": result['type'],
            "metadata": {
                "chapter": chapter,
                "query": query,
                "num_results": len(result['results'])
            }
        }
    except Exception as e:
        return {
            "answer": f"Error processing query: {str(e)}",
            "type": "error",
            "metadata": {"error": str(e)}
        }