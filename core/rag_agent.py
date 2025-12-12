from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
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

def classify_exercise_intent(query: str, exercise_id: str) -> str:
    """
    Uses LLM to decide exactly what the user wants regarding a specific exercise.
    Returns: 'get_question', 'get_answer', or 'explain_solution'
    """
    # JsonOutputParser: Parses the LLM's string output into a Python dictionary.
    parser = JsonOutputParser()
    
    prompt = ChatPromptTemplate.from_template("""
    You are a query analyzer for a Statistics Textbook.
    The user is asking about Exercise {exercise_id}.
    
    Classify their intent into ONE category:
    1. "get_question": User wants to read/see the exercise text (e.g., "what is 6.9?", "read me 6.9")
    2. "get_answer": User JUST wants the final number/key (e.g., "answer to 6.9", "is 6.9 a or b?")
    3. "explain_solution": User wants to know HOW to solve it (e.g., "how do I do 6.9?", "solution for 6.9", "help with 6.9")

    USER QUERY: "{query}"

    Return JSON only:
    {{ "intent": "get_question" | "get_answer" | "explain_solution" }}
    """)

    try:
        chain = prompt | llm | parser
        result = chain.invoke({"query": query, "exercise_id": exercise_id})
        return result.get("intent", "explain_solution")
    except Exception as e:
        print(f"[Intent Error] {e}")
        return "explain_solution" # Fallback to explanation if uncertain


# =====================================================
#  Targeted Response Generation
# =====================================================

def generate_targeted_response(intent: str, query: str, data: dict) -> str:
    """
    Generates a specific, professional response based on the intent.
    This replaces run_rag for direct queries to avoid 'meta-talk'.
    """
    
    # 1. Prompt for getting the Answer Key only
    if intent == "get_answer":
        template = """
        You are a strict Answer Key Assistant.
        Task: Provide the official answer to the exercise based ONLY on the context.
        
        Rules:
        - Output the final value/answer clearly.
        - Use LaTeX in Markdown for any mathematical expressions (inline with $...$, block with $$...$$).
        - Do NOT explain the steps unless the answer key is missing.
        - Do NOT say "The student is asking..." or "Here is the answer". Just give the answer.
        
        Context (Official Answer):
        {answer_text}
        
        User Question: {query}
        """

    # 2. Prompt for getting the Question Text only
    elif intent == "get_question":
        template = """
        You are a Textbook Reader.
        Task: Output the exact text of the requested exercise.
        
        Rules:
        - Quote the exercise text exactly as it appears in the context.
        - If there is a formula, rewrite it in LaTeX in Markdown (use $...$ or $$...$$) so it renders cleanly.
        - Do NOT solve it.
        
        Context (Exercise Text):
        {exercise_text}
        
        User Question: {query}
        """


    # 3. Prompt for Explanation (The detailed tutor)
    else:  # explain_solution
        template = """
        You are a Professional Statistics Tutor.
        Task: Explain the solution to the exercise step-by-step.
        
        Context:
        - Exercise: {exercise_text}
        - Official Answer: {answer_text}
        - Relevant Theory: {theory_text}
        
        Rules:
        - Start directly with the solution steps.
        - Do NOT say "The student is asking about...".
        - Use the Official Answer to verify your logic, but show the work to get there.
        - Use LaTeX for math where appropriate (inline with $...$, block with $$...$$).
        
        User Question: {query}
        """


    prompt = ChatPromptTemplate.from_template(template)
    
    # StrOutputParser: Converts the LLM response object directly into a clean string.
    output_parser = StrOutputParser()
    
    chain = prompt | llm | output_parser
    
    return chain.invoke({
        "query": query,
        "exercise_text": data.get("exercise_text", ""),
        "answer_text": data.get("answer_text", ""),
        "theory_text": data.get("theory_text", "")
    })

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

CHAT HISTORY:
{chat_history}
                                            
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
11. When you write mathematical expressions in the Final Answer, format them as LaTeX in Markdown ($...$ for inline, $$...$$ for block equations). Rewrite messy textbook formulas into clean LaTeX.
12. The content under "Thought", "Action", and "Observation" is for your internal reasoning only. The text after "Final Answer:" must be a clean explanation that can be shown directly to the student.

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
def normalize_latex_for_markdown(text: str) -> str:
    r"""
    Convert TeX environments into Markdown math so remark-math + KaTeX can render them.

    - \begin{equation*}...\end{equation*}  →  $$ ... $$
    - \[ ... \]                              →  $$ ... $$
    - Any other \begin{X}...\end{X} block   →  $$ ... $$
      (e.g. \begin{cases}...\end{cases})
    """

    # 1) \begin{equation}...\end{equation} → $$ ... $$
    text = re.sub(
        r"\\begin{equation\*?}(.*?)\\end{equation\*?}",
        lambda m: f"$$ {m.group(1).strip()} $$",
        text,
        flags=re.DOTALL,
    )

    # 2) \[ ... \] → $$ ... $$
    text = re.sub(
        r"\\\[(.*?)\\\]",
        lambda m: f"$$ {m.group(1).strip()} $$",
        text,
        flags=re.DOTALL,
    )

    # 3) Any other LaTeX environment: \begin{X}...\end{X} → $$ ... $$
    #    (this will catch things like \begin{cases}...\end{cases})
    text = re.sub(
        r"\\begin{([^}]+)}(.*?)\\end{\1}",
        lambda m: f"$$\n\\begin{{{m.group(1)}}}{m.group(2)}\\end{{{m.group(1)}}}\n$$",
        text,
        flags=re.DOTALL,
    )

    return text

def format_lettered_subparts(text: str) -> str:
    """
    Ensure exercise sub-parts (a), (b), (c), (d) start on new lines.

    Turns:
        "... = 1. (a) Show that ..." 
    into:
        "... = 1.

        (a) Show that ..."
    """
    # space + (a|b|c|d) + space  ->  two newlines + (a|b|c|d) + space
    return re.sub(r"\s\(([a-d])\)\s", r"\n\n(\1) ", text)

def ask_agent(query: str, chapter: int = None, chat_history: list = []) -> dict:
    """
    Main interface to ask the AI Agent.
    """
    try:
        enhanced_query = query
        if chapter:
            enhanced_query = f"[From Chapter {chapter}] {query}"
        
        print(f"[AGENT] Processing: {enhanced_query}")
        
        history_str = ""
        if chat_history:
            history_str = "\n".join([f"{role.title()}: {msg}" for role, msg in chat_history])

        # Execute agent
        result = agent_executor.invoke({"input": enhanced_query, "chat_history": history_str})
        clean_answer = normalize_latex_for_markdown(result["output"])
        clean_answer = format_lettered_subparts(clean_answer)

        tools_used = [
            step[0].tool for step in result.get("intermediate_steps", [])
            if hasattr(step[0], 'tool')
        ]
        
        return {
            "answer": clean_answer,
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
def ask_direct(query: str, chapter: int = None, chat_history: list = []) -> dict: # <--- Added history
    """
    Direct retrieval with Intent Detection.
    """
    try:
        print(f"[DIRECT] Processing: {query}")
        
        id_match = re.search(r'(\d+\.\d+)', query)
        
        if id_match:
            ex_id = id_match.group(1)
            print(f"[DIRECT] Detected ID {ex_id}. Determining specific intent...")
            intent = classify_exercise_intent(query, ex_id)
            print(f"[DIRECT] Intent detected: {intent}")

            exercise_docs = get_exercise(ex_id, chapter)
            exercise_text = exercise_docs[0].page_content if exercise_docs else f"Text for {ex_id} not found."
            
            answer_docs = get_answer(ex_id, chapter)
            answer_text = answer_docs[0].page_content if answer_docs else "Answer key not found."
            
            theory_text = ""
            if intent == "explain_solution":
                theory_docs = get_theory_concepts(exercise_text, chapter, limit=2)
                theory_text = "\n\n".join([d.page_content for d in theory_docs])

            data_context = {"exercise_text": exercise_text, "answer_text": answer_text, "theory_text": theory_text}
            
            final_answer = generate_targeted_response(intent, query, data_context)
            final_answer = normalize_latex_for_markdown(final_answer)
            final_answer = format_lettered_subparts(final_answer)
            
            return {
                "answer": final_answer,
                "type": f"targeted_{intent}",
                "metadata": {"chapter": chapter, "exercise_id": ex_id, "intent": intent, "success": True}
            }

        else:
            print("[DIRECT] No ID detected. Running semantic search.")
            result = smart_search(query, chapter)
            
            if result['results']:
                # Pass chat_history to run_rag
                answer = run_rag(query, result['results'], chat_history=chat_history) # <--- Pass history
            else:
                answer = "I couldn't find relevant information in the textbook for that query."
            
            answer = normalize_latex_for_markdown(answer)
            answer = format_lettered_subparts(answer)

            return {
                "answer": answer,
                "type": result.get('type', 'general'),
                "metadata": {"chapter": chapter, "num_results": len(result.get('results', [])), "success": True}
            }

    except Exception as e:
        print(f"[DIRECT ERROR] {str(e)}")
        return {
            "answer": f"Error processing query: {str(e)}",
            "type": "error",
            "metadata": {"error": str(e), "success": False}
        }