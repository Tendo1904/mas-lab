from langchain_openai import OpenAI
from state_types import GraphState, Classification, Plan
from utils import keyword_search_notes
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen")

def make_llm():
    return OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY, name=MODEL_NAME, temperature=0.2, max_tokens=1024)

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