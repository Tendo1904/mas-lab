from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from state_types import GraphState, Classification, Plan
from utils import keyword_search_notes, append_note
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen")

def make_llm():
    print(MODEL_NAME)
    return ChatOpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY, model=MODEL_NAME, temperature=0.2, max_tokens=1024)

def router_node(state: GraphState) -> GraphState:
    state.add_agent("router")
    q = state.query.lower()
    if any(w in q for w in ["code", "implement", "function", "python", "javascript"]):
        state.classification = Classification(type="code")
    elif any(w in q for w in ["architecture", "design", "pattern", "mas"]):
        state.classification = Classification(type="architecture")
    elif any(w in q for w in ["what is", "explain", "theory", "define"]):
        state.classification = Classification(type="conceptual")
    else:
        state.classification = Classification(type="general")
    return state

def planner_node(state: GraphState) -> GraphState:
    state.add_agent("planner")
    notes = keyword_search_notes(state.query)

    p = Plan()
    p.context_notes = notes
    ctype = state.classification.type if state.classification else "general"
    if ctype == "code":
        p.steps = ["gather_context", "generate_code", "run_unit_tests", "format_answer"]
        p.tools = ["CodeHelper", "TestRunner"]
    elif ctype == "architecture":
        p.steps = ["gather_context", "generate_architecture_design", "format_answer"]
        p.tools = ["RAGRetriever"]
    else:
        p.steps = ["gather_context", "generate_answer", "format_answer"]
        p.tools = ["RAGRetriever"]
    state.plan = p
    return state

def rag_retriever_node(state: GraphState) -> GraphState:
    state.add_agent("rag_retriever")
    notes = keyword_search_notes(state.query)
    if notes:
        state.partial_answers.rag_context = "\n\n".join([n.text for n in notes])
    return state

def executor_node(state: GraphState) -> GraphState:
    state.add_agent("executor")
    llm = make_llm()
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
        HumanMessagePromptTemplate.from_template("Query: {query}\n\nContext: {context}") 
    ])
    messages = prompt.format_prompt(**{"query": state.query, "context": state.partial_answers.rag_context or ""}).to_messages()
    try:
        gen = llm.generate([messages])
        text = gen.generations[0][0].text if hasattr(gen, "generations") else str(gen)
    except Exception as e:
        print(e)
    state.partial_answers.executor_result = text
    return state

def formatter_node(state: GraphState) -> GraphState:
    state.add_agent("formatter")
    parts = []
    if state.partial_answers.executor_result:
        parts.append(state.partial_answers.executor_result)
    if state.partial_answers.rag_context:
        parts.append("\n\n---\nContext used:\n" + state.partial_answers.rag_context)
    final = "\n\n".join(parts) if parts else "Beg a pardon, ain`t got anything."
    state.final_answer = final
    state.add_session_entry(question=state.query, answer=final)
    append_note(f"QA: {state.query} -> {final[:200]}", tags=["auto"])
    return state

def supervisor_node(state: GraphState) -> GraphState:
    state.add_agent("supervisor")
    fa = state.final_answer or ""
    if "forbidden" in fa.lower():
        state.final_answer = "Query was rejected due to safety policies."
    return state